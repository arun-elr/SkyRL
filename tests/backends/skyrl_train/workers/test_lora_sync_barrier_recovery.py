"""Tests for distributed barrier recovery during LoRA / weight sync failures.

Covers issue #1827: when rank 0 (or a writer rank) raises inside
``_save_lora_adapters_and_sync``, ``_publish_lora_adapter_in_memory``, or
prefix-cache reset before a ``torch.distributed.barrier()``, peer ranks must
not hang until Gloo times out and poisons the process group for subsequent
multi-tenant ``save_weights_for_sampler`` requests.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    InferenceServerTimeoutError,
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import (
    PeerRankSyncError,
    barrier_with_error_propagation,
    cleanup_unawaited_task,
)


class _FakeTaskType(Enum):
    CAUSAL_LM = "CAUSAL_LM"


class _FakePeftType(Enum):
    LORA = "LORA"


@dataclass
class _FakePeftConfig:
    r: int
    lora_alpha: int
    target_modules: tuple
    task_type: _FakeTaskType
    peft_type: _FakePeftType


def _init_gloo_group(rank: int, world_size: int, store_path: str, timeout_s: float = 5.0) -> None:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )


def _run_mp(fn, world_size: int, tmp_path: Path, *extra_args) -> None:
    store_path = str(tmp_path / "dist_store")
    mp.spawn(fn, args=(world_size, store_path, *extra_args), nprocs=world_size, join=True)


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: barrier_with_error_propagation
# ---------------------------------------------------------------------------


def _worker_barrier_rank0_failure_and_recovery(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    try:
        # Round 1: rank 0 fails with a connection error before the barrier.
        local_error = RuntimeError("ClientConnectorError: Cannot connect to vLLM server") if rank == 0 else None
        if rank == 0:
            with pytest.raises(RuntimeError, match="Cannot connect to vLLM server"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter load")
        else:
            with pytest.raises(PeerRankSyncError, match=r"Rank 0 failed during LoRA adapter load"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter load")

        # Rounds 2 and 3: process group must still be healthy (no pair closure / timeout).
        barrier_with_error_propagation(None, stage="LoRA adapter load")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_barrier_with_error_propagation_rank0_failure_and_recovery(tmp_path):
    _run_mp(_worker_barrier_rank0_failure_and_recovery, 4, tmp_path)


def _worker_barrier_nonzero_rank_failure(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    try:
        # Simulate a non-zero writer rank (e.g. second node leader) hitting disk full.
        local_error = OSError("No space left on device") if rank == 2 else None
        if rank == 2:
            with pytest.raises(OSError, match="No space left on device"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter export")
        else:
            with pytest.raises(PeerRankSyncError, match=r"Rank 2 failed during LoRA adapter export"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter export")

        # Subsequent sync succeeds across all 4 ranks.
        barrier_with_error_propagation(None, stage="LoRA adapter export")
    finally:
        dist.destroy_process_group()


def test_barrier_with_error_propagation_nonzero_rank_failure(tmp_path):
    _run_mp(_worker_barrier_nonzero_rank_failure, 4, tmp_path)


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: FSDP LoRA sync + prefix cache reset recovery
# ---------------------------------------------------------------------------


def _make_fsdp_lora_worker(base_sync_path: str) -> FSDPPolicyWorkerBase:
    peft_cfg = _FakePeftConfig(
        r=4,
        lora_alpha=8,
        target_modules=("q_proj",),
        task_type=_FakeTaskType.CAUSAL_LM,
        peft_type=_FakePeftType.LORA,
    )
    peft_model = SimpleNamespace(peft_config={"default": peft_cfg})
    worker = FSDPPolicyWorkerBase.__new__(FSDPPolicyWorkerBase)
    worker._is_lora = True
    worker._is_multimodal_lm_only = False
    worker._weight_sync_inference_client = None
    worker.model = SimpleNamespace(model=peft_model)
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(
            model=SimpleNamespace(lora=SimpleNamespace(lora_sync_path=base_sync_path)),
        ),
    )
    return worker


def _worker_fsdp_lora_load_failure_recovers(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    from skyrl.backends.skyrl_train.distributed import fsdp_utils

    _init_gloo_group(rank, world_size, store_path)
    orig_collect = fsdp_utils.collect_lora_params
    orig_empty_cache = torch.cuda.empty_cache
    try:
        fsdp_utils.collect_lora_params = lambda module: {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        }
        torch.cuda.empty_cache = lambda: None

        worker = _make_fsdp_lora_worker(sync_dir)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        loaded = []
        resets = []

        async def _load_lora(name: str, path: str):
            if name == "tenant-a":
                raise RuntimeError("Failed to load LoRA adapter 'tenant-a': ClientConnectorError")
            loaded.append((name, path))

        async def _reset_cache(reset_running_requests: bool = True):
            resets.append(reset_running_requests)

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock(side_effect=_load_lora)
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)

        async def _run():
            # Request 1 fails on rank 0 during load_lora_adapter; rank 1 must exit cleanly.
            if rank == 0:
                with pytest.raises(RuntimeError, match="ClientConnectorError"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

            # Requests 2 & 3 for other tenants must succeed without barrier timeouts.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-c")

        asyncio.run(_run())

        if rank == 0:
            assert [name for name, _ in loaded] == ["tenant-b", "tenant-c"]
            assert len(resets) == 2
            assert (Path(sync_dir) / "tenant-b" / "adapter_model.safetensors").exists()
            assert (Path(sync_dir) / "tenant-c" / "adapter_model.safetensors").exists()
        else:
            assert loaded == []
            assert resets == []
    finally:
        fsdp_utils.collect_lora_params = orig_collect
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


def test_fsdp_lora_sync_load_failure_recovers_for_subsequent_tenants(tmp_path):
    _run_mp(_worker_fsdp_lora_load_failure_recovers, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_fsdp_prefix_cache_reset_failure_recovers(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    from skyrl.backends.skyrl_train.distributed import fsdp_utils

    _init_gloo_group(rank, world_size, store_path)
    orig_collect = fsdp_utils.collect_lora_params
    orig_empty_cache = torch.cuda.empty_cache
    try:
        fsdp_utils.collect_lora_params = lambda module: {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
        }
        torch.cuda.empty_cache = lambda: None

        worker = _make_fsdp_lora_worker(sync_dir)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        reset_attempts = 0

        async def _reset_cache(reset_running_requests: bool = True):
            nonlocal reset_attempts
            reset_attempts += 1
            if reset_attempts == 1:
                raise RuntimeError("Prefix cache reset failed on http://vllm:8001")

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock()
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)

        async def _run():
            if rank == 0:
                with pytest.raises(RuntimeError, match="Prefix cache reset failed"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")

        asyncio.run(_run())
        if rank == 0:
            assert reset_attempts == 2
    finally:
        fsdp_utils.collect_lora_params = orig_collect
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


def test_fsdp_lora_sync_prefix_cache_reset_failure_recovers(tmp_path):
    _run_mp(_worker_fsdp_prefix_cache_reset_failure_recovers, 2, tmp_path, str(tmp_path / "lora_sync"))


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: Worker._sync_weights_to_inference_engines
# ---------------------------------------------------------------------------


def _worker_full_weight_sync_prefix_cache_failure_recovers(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _Engine:
            skyrl_handles_prefix_cache_reset = False
            skyrl_force_disable_expandable_segments = False
            skyrl_empty_cache_after_send = False

            def __init__(self):
                self.sends = 0

            def send_weights(self):
                self.sends += 1

        worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
        worker._weight_sync_engine = _Engine()
        worker.cfg = SimpleNamespace(
            fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
            placement=SimpleNamespace(colocate_all=False),
        )

        @contextmanager
        def _noop_ctx(*args, **kwargs):
            yield

        worker._expandable_segments_disabled_for_sync = _noop_ctx

        reset_attempts = 0

        async def _reset_cache(reset_running_requests: bool = True):
            nonlocal reset_attempts
            reset_attempts += 1
            if reset_attempts == 1:
                raise RuntimeError("vLLM unreachable during reset_prefix_cache")

        client = Mock()
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        async def _run():
            if rank == 0:
                with pytest.raises(RuntimeError, match="vLLM unreachable"):
                    await worker._sync_weights_to_inference_engines(client, ie_cfg)
            else:
                await worker._sync_weights_to_inference_engines(client, ie_cfg)

            await worker._sync_weights_to_inference_engines(client, ie_cfg)

        asyncio.run(_run())
        assert worker._weight_sync_engine.sends == 2
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.vllm
def test_sync_weights_prefix_cache_reset_failure_recovers(tmp_path):
    _run_mp(_worker_full_weight_sync_prefix_cache_failure_recovers, 2, tmp_path)


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: Megatron LoRA disk & in-memory sync recovery
# ---------------------------------------------------------------------------


def _make_megatron_lora_worker(base_sync_path: str, *, sync_mode: str = "disk"):
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronPolicyWorkerBase,
    )

    worker = MegatronPolicyWorkerBase.__new__(MegatronPolicyWorkerBase)
    worker._is_lora = True
    worker._weight_sync_inference_client = None
    worker._weight_sync_engine = None
    worker._lora_weight_source = None
    worker._logical_model_path = "/models/base"
    worker.actor_module = []
    worker.provider = SimpleNamespace(num_moe_experts=None)
    worker.lora_cls = SimpleNamespace(dim=4, alpha=8, target_modules=["q_proj"])
    worker.bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(model_name_or_path="/models/base"),
        export_adapter_weights=lambda actor_module, cpu=True, show_progress=False: [
            ("layers.0.self_attn.q_proj.lora_A.weight", torch.ones(4, 8, dtype=torch.bfloat16)),
            ("layers.0.self_attn.q_proj.lora_B.weight", torch.ones(8, 4, dtype=torch.bfloat16)),
        ],
        _model_bridge=SimpleNamespace(_adapter_target_modules=lambda names: ["q_proj"]),
    )
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(
            model=SimpleNamespace(lora=SimpleNamespace(lora_sync_path=base_sync_path, sync_mode=sync_mode)),
            megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False)),
        ),
    )

    @contextmanager
    def _noop_ctx(*args, **kwargs):
        yield

    worker._expandable_segments_disabled_for_sync = _noop_ctx
    return worker


def _worker_megatron_disk_load_failure_recovers(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    import megatron.bridge.models.conversion.peft_bridge as peft_bridge

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    orig_build_cfg = peft_bridge.build_adapter_config_dict
    try:
        torch.cuda.empty_cache = lambda: None
        peft_bridge.build_adapter_config_dict = lambda lora_cls, target_modules, base_model_name_or_path: {
            "r": lora_cls.dim,
            "lora_alpha": lora_cls.alpha,
            "target_modules": list(target_modules),
            "base_model_name_or_path": base_model_name_or_path,
        }

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="disk")
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        loaded = []
        resets = []

        async def _load_lora(name: str, path: str):
            if name == "model_1":
                raise RuntimeError("Failed to load LoRA adapter 'model_1' on 1/4 server(s): ClientConnectorError")
            loaded.append((name, path))

        async def _reset_cache(reset_running_requests: bool = True):
            resets.append(reset_running_requests)

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock(side_effect=_load_lora)
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)

        async def _run():
            # Request 1 fails during rank-0 load_lora_adapter.
            if rank == 0:
                with pytest.raises(RuntimeError, match="ClientConnectorError"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")

            # Subsequent multi-tenant requests must succeed without barrier timeouts.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_2")
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_3")

        asyncio.run(_run())

        if rank == 0:
            assert [name for name, _ in loaded] == ["model_2", "model_3"]
            assert len(resets) == 2
        else:
            assert loaded == []
            assert resets == []
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        peft_bridge.build_adapter_config_dict = orig_build_cfg
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_lora_disk_sync_load_failure_recovers_multi_tenant(tmp_path):
    _run_mp(_worker_megatron_disk_load_failure_recovers, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_megatron_disk_writer_failure_skips_load(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    import megatron.bridge.models.conversion.peft_bridge as peft_bridge

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    orig_build_cfg = peft_bridge.build_adapter_config_dict
    try:
        torch.cuda.empty_cache = lambda: None
        fail_first = {"flag": True}

        def _build_cfg(lora_cls, target_modules, base_model_name_or_path):
            if fail_first["flag"]:
                fail_first["flag"] = False
                raise OSError("disk full while exporting adapter")
            return {
                "r": lora_cls.dim,
                "lora_alpha": lora_cls.alpha,
                "target_modules": list(target_modules),
                "base_model_name_or_path": base_model_name_or_path,
            }

        peft_bridge.build_adapter_config_dict = _build_cfg

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="disk")
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock()

        async def _run():
            if rank == 0:
                with pytest.raises(OSError, match="disk full"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")
                client.load_lora_adapter.assert_not_awaited()
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")

            # Next request succeeds on both ranks.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_2")
            if rank == 0:
                client.load_lora_adapter.assert_awaited_once()

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        peft_bridge.build_adapter_config_dict = orig_build_cfg
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_lora_disk_sync_writer_failure_skips_load_and_recovers(tmp_path):
    _run_mp(_worker_megatron_disk_writer_failure_skips_load, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_megatron_memory_sync_failures_recover(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _FakeLoraSource(LoraAdapterWeightSource):
            def __init__(self):
                super().__init__(dtype=torch.bfloat16, experts_per_shared_adapter=1)
                self.prepare_calls = []

            def export_adapter_stream(self):
                self.prepare_calls.append(self._lora_name)
                if self._lora_name == "tenant-prep-fail" and rank == 1:
                    raise RuntimeError("rank-1 prepare failure")
                return [("base_model.model.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.bfloat16))]

        class _FakeEngine:
            skyrl_force_disable_expandable_segments = False

            def __init__(self, source):
                self.source = source
                self.sends = []

            def send_weights(self):
                self.sends.append(self.source._lora_name)
                list(self.source.iter_full())

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="memory")
        source = _FakeLoraSource()
        engine = _FakeEngine(source)
        worker._lora_weight_source = source
        worker._weight_sync_engine = engine
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)

        async def _set_target(target):
            if target["lora_name"] == "tenant-a":
                raise RuntimeError("Cannot connect to server during set_lora_receive_target")

        async def _load_lora(lora_name: str, in_memory: bool = False):
            assert in_memory is True
            if lora_name == "tenant-b":
                raise RuntimeError("Cannot connect to server during in-memory load_lora_adapter")

        client = Mock(spec=RemoteInferenceClient)
        client.set_lora_receive_target = AsyncMock(side_effect=_set_target)
        client.load_lora_adapter = AsyncMock(side_effect=_load_lora)

        async def _run():
            # Request 0: rank-1 fails during source.prepare(); rank 0 must NOT arm receivers.
            if rank == 1:
                with pytest.raises(RuntimeError, match="rank-1 prepare failure"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-prep-fail")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-prep-fail")
                client.set_lora_receive_target.assert_not_awaited()
            assert engine.sends == []
            assert source._prepared is None

            # Request 1: set_lora_receive_target fails on rank 0.
            # Neither rank may enter send_weights(), and prepared state must be discarded.
            if rank == 0:
                with pytest.raises(RuntimeError, match="set_lora_receive_target"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            assert engine.sends == []
            assert source._prepared is None

            # Request 2: send_weights succeeds, then load_lora_adapter fails on rank 0.
            if rank == 0:
                with pytest.raises(RuntimeError, match="in-memory load_lora_adapter"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            assert engine.sends == ["tenant-b"]

            # Request 3: full in-memory sync succeeds with fresh export for tenant-c.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-c")
            assert engine.sends == ["tenant-b", "tenant-c"]
            assert source.prepare_calls == ["tenant-prep-fail", "tenant-a", "tenant-b", "tenant-c"]

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_lora_memory_sync_arm_and_load_failures_recover(tmp_path):
    _run_mp(_worker_megatron_memory_sync_failures_recover, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_megatron_probe_failure_does_not_strand_peers(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    _init_gloo_group(rank, world_size, store_path)
    try:
        # Use a regular file path as base_sync_path so rank 0's os.makedirs fails with FileExistsError/NotADirectoryError.
        bad_path = str(Path(sync_dir) / "not_a_dir")
        if rank == 0:
            Path(sync_dir).mkdir(parents=True, exist_ok=True)
            Path(bad_path).write_text("blocker", encoding="utf-8")
        dist.barrier()

        worker = _make_megatron_lora_worker(bad_path, sync_mode="disk")
        if rank == 0:
            with pytest.raises(OSError):
                worker._is_lora_sync_writer_rank()
        else:
            with pytest.raises(PeerRankSyncError, match="writer probe"):
                worker._is_lora_sync_writer_rank()

        # Once pointed at a valid directory, probe succeeds on both ranks.
        worker.cfg.policy.model.lora.lora_sync_path = str(Path(sync_dir) / "valid_dir")
        is_writer = worker._is_lora_sync_writer_rank()
        assert is_writer == (rank == 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_writer_rank_probe_failure_does_not_strand_peers(tmp_path):
    _run_mp(_worker_megatron_probe_failure_does_not_strand_peers, 2, tmp_path, str(tmp_path / "probe"))


# ---------------------------------------------------------------------------
# Fast in-process unit & concurrency tests
# ---------------------------------------------------------------------------


def test_barrier_with_error_propagation_uninitialized_falls_back_to_barrier(monkeypatch):
    barriers = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "barrier", lambda: barriers.append(1))

    barrier_with_error_propagation(None, stage="test")
    assert len(barriers) == 1

    with pytest.raises(ValueError, match="boom"):
        barrier_with_error_propagation(ValueError("boom"), stage="test")
    assert len(barriers) == 2


@pytest.mark.asyncio
async def test_cleanup_unawaited_task_handles_coro_and_task():
    async def _never_awaited():
        return 1

    coro = _never_awaited()
    await cleanup_unawaited_task(coro)
    assert coro.cr_frame is None

    cancelled_cleanly = False

    async def _background_task():
        nonlocal cancelled_cleanly
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled_cleanly = True
            raise

    task = asyncio.create_task(_background_task())
    await asyncio.sleep(0)
    await cleanup_unawaited_task(task)
    assert task.cancelled()
    assert cancelled_cleanly is True

    loop = asyncio.get_running_loop()
    failed_fut = loop.create_future()
    failed_fut.set_exception(RuntimeError("already failed"))
    await cleanup_unawaited_task(failed_fut)
    assert isinstance(failed_fut.exception(), RuntimeError)


@pytest.mark.vllm
def test_lora_source_discard_prepared_forces_reexport():
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource

    class _CountingSource(LoraAdapterWeightSource):
        def __init__(self):
            super().__init__(dtype=torch.float32, experts_per_shared_adapter=1)
            self.exports = 0

        def export_adapter_stream(self):
            self.exports += 1
            return [("base_model.model.q_proj.lora_A.weight", torch.full((2, 2), float(self.exports)))]

    source = _CountingSource()
    source.set_lora_name("tenant-a")
    source.prepare()
    assert source.exports == 1
    assert len(source.metadata) == 1

    # Discarding clears both _prepared and cached metadata so the next tenant re-exports.
    source.discard_prepared()
    source.set_lora_name("tenant-b")
    source.prepare()
    assert source.exports == 2
    assert source.receive_target["lora_name"] == "tenant-b"


@pytest.mark.asyncio
async def test_await_weight_sync_thread_drains_on_cancellation_before_discard():
    """Cancellation must wait for the worker thread before finally clears prepared state."""
    import threading
    import time

    worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
    started = threading.Event()
    state = {"prepared": None, "thread_done": False}

    def _slow_prepare():
        started.set()
        time.sleep(0.15)
        state["prepared"] = "stale-tenant-a"
        state["thread_done"] = True

    async def _run_with_cleanup():
        try:
            await worker._await_weight_sync_thread(_slow_prepare)
        finally:
            # Must observe thread_done=True so _slow_prepare cannot repopulate after discard.
            assert state["thread_done"] is True
            state["prepared"] = None

    task = asyncio.create_task(_run_with_cleanup())
    while not started.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["thread_done"] is True
    assert state["prepared"] is None


def _worker_full_weight_sync_send_failure_on_rank0_recovers(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _Engine:
            skyrl_handles_prefix_cache_reset = False
            skyrl_force_disable_expandable_segments = False
            skyrl_empty_cache_after_send = False

            def __init__(self):
                self.sends = 0

            def send_weights(self):
                self.sends += 1
                if rank == 0 and self.sends == 1:
                    raise RuntimeError("rank-0 HTTP finish RPC failed after collectives")

        worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
        worker._weight_sync_engine = _Engine()
        worker.cfg = SimpleNamespace(
            fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
            placement=SimpleNamespace(colocate_all=False),
        )

        @contextmanager
        def _noop_ctx(*args, **kwargs):
            yield

        worker._expandable_segments_disabled_for_sync = _noop_ctx

        resets = []

        async def _reset_cache(reset_running_requests: bool = True):
            resets.append(reset_running_requests)

        client = Mock()
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        async def _run():
            if rank == 0:
                with pytest.raises(RuntimeError, match="rank-0 HTTP finish RPC failed"):
                    await worker._sync_weights_to_inference_engines(client, ie_cfg)
            else:
                await worker._sync_weights_to_inference_engines(client, ie_cfg)

            # Second sync must succeed cleanly on both ranks.
            await worker._sync_weights_to_inference_engines(client, ie_cfg)

        asyncio.run(_run())
        assert worker._weight_sync_engine.sends == 2
        if rank == 0:
            assert len(resets) == 1
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.vllm
def test_sync_weights_send_failure_on_rank0_recovers(tmp_path):
    _run_mp(_worker_full_weight_sync_send_failure_on_rank0_recovers, 2, tmp_path)


@pytest.mark.asyncio
async def test_set_lora_receive_target_settles_and_disarms_on_partial_failure():
    """Partial arm failure (including error-after-arm) settles and disarms all servers."""
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-lost-resp:8000", "http://server-good:8001"]

    armed = {}
    calls = []

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        calls.append((url, method_name))
        if method_name == "skyrl_set_lora_receive_target":
            if "server-lost-resp" in url:
                # Server armed locally before losing the HTTP response.
                armed[url] = json["kwargs"]["receive_target"]["lora_name"]
                raise ConnectionResetError("connection lost after arm")
            await asyncio.sleep(0.08)
            if url in armed:
                raise RuntimeError(f"already armed for {armed[url]}")
            armed[url] = json["kwargs"]["receive_target"]["lora_name"]
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            armed.pop(url, None)
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    with pytest.raises(ConnectionResetError, match="connection lost after arm"):
        await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"})

    # Both servers (including the one whose HTTP response failed) must be disarmed.
    assert armed == {}
    assert ("http://server-lost-resp:8000", "skyrl_clear_lora_receive_target") in calls
    assert ("http://server-good:8001", "skyrl_clear_lora_receive_target") in calls

    # Next tenant can arm both servers cleanly.
    client.server_urls = ["http://server-good:8001"]
    await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
    assert armed == {"http://server-good:8001": "tenant-b"}


@pytest.mark.asyncio
async def test_set_lora_receive_target_cancellation_drains_and_disarms():
    """Parent cancellation drains in-flight arm RPCs and disarms all servers."""
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-fast:8000", "http://server-slow:8001"]

    armed = {}
    fast_armed = asyncio.Event()

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        if method_name == "skyrl_set_lora_receive_target":
            if "server-fast" in url:
                armed[url] = json["kwargs"]["receive_target"]["lora_name"]
                fast_armed.set()
                return url, {"status": 200, "body": {}}
            await asyncio.sleep(0.1)
            armed[url] = json["kwargs"]["receive_target"]["lora_name"]
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            armed.pop(url, None)
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    task = asyncio.create_task(client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"}))
    await fast_armed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert armed == {}


@pytest.mark.vllm
def test_skyrl_clear_lora_receive_target_allows_rearm():
    from skyrl.backends.skyrl_train.weight_sync.lora_target import (
        build_lora_receive_target,
    )
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    class _DummyReceiver(SkyrlLoraStagingMixin):
        vllm_config = SimpleNamespace(lora_config=object())

    receiver = _DummyReceiver()
    receiver.skyrl_set_lora_receive_target(build_lora_receive_target("tenant-a", {}, {}))
    assert receiver.skyrl_lora_armed() is True

    receiver.skyrl_clear_lora_receive_target()
    assert receiver.skyrl_lora_armed() is False

    # Re-arming for the next tenant succeeds cleanly.
    receiver.skyrl_set_lora_receive_target(build_lora_receive_target("tenant-b", {}, {}))
    assert receiver.skyrl_lora_armed() is True


@pytest.mark.asyncio
async def test_load_lora_adapter_settles_all_servers_before_raising(monkeypatch):
    """When one server fails fast, load_lora_adapter settles sibling loads before raising."""
    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as ric_mod,
    )

    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-bad:8000", "http://server-slow:8001"]
    client._get_session = AsyncMock(return_value=object())

    slow_settled = False

    class _FakeResp:
        status = 200

        async def text(self):
            return "Success"

    class _FakeRequestCtx:
        def __init__(self, url: str):
            self.url = url

        async def __aenter__(self):
            nonlocal slow_settled
            if "server-bad" in self.url:
                raise ConnectionRefusedError("Cannot connect to server-bad")
            await asyncio.sleep(0.1)
            slow_settled = True
            return _FakeResp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ric_mod, "_control_plane_request", lambda session, method, url, **kw: _FakeRequestCtx(url))

    with pytest.raises(ConnectionRefusedError, match="server-bad"):
        await client.load_lora_adapter("tenant-a", "/tmp/lora/tenant-a")

    assert slow_settled is True


# ---------------------------------------------------------------------------
# Regression tests for bounded phase timeout, send-failure disarm, and dirty-server recovery
# ---------------------------------------------------------------------------


def _worker_stalled_arm_and_cleanup_bounded_below_gloo_timeout(rank: int, world_size: int, store_path: str) -> None:
    import time

    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as ric_mod,
    )

    pg_timeout_s = 2.5
    _init_gloo_group(rank, world_size, store_path, timeout_s=pg_timeout_s)
    orig_pg_timeout = ric_mod.SKYRL_WORKER_NCCL_TIMEOUT_IN_S
    try:
        ric_mod.SKYRL_WORKER_NCCL_TIMEOUT_IN_S = pg_timeout_s

        client = RemoteInferenceClient.__new__(RemoteInferenceClient)
        client.server_urls = ["http://server-stalled:8000", "http://server-ok:8001"]
        stalled = {"flag": True}
        armed = {}

        async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
            method_name = json["method"]
            if "server-stalled" in url and stalled["flag"]:
                # Both arm and rollback stall far beyond the 2.5s Gloo PG timeout.
                await asyncio.sleep(30.0)
            if method_name == "skyrl_set_lora_receive_target":
                if url in armed:
                    raise RuntimeError(f"already armed for {armed[url]}")
                armed[url] = json["kwargs"]["receive_target"]["lora_name"]
                return url, {"status": 200, "body": {}}
            if method_name == "skyrl_clear_lora_receive_target":
                armed.pop(url, None)
                return url, {"status": 200, "body": {}}
            raise AssertionError(f"Unexpected method {method_name}")

        client._call_server = _fake_call_server

        async def _run():
            started = time.monotonic()
            arm_error = None
            if rank == 0:
                try:
                    await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"})
                except Exception as exc:
                    arm_error = exc
            if rank == 0:
                with pytest.raises(InferenceServerTimeoutError) as exc_info:
                    barrier_with_error_propagation(arm_error, stage="in-memory LoRA receive target setup")
                # Must finish strictly below the 2.5s Gloo timeout AND note the rollback timeout.
                elapsed = time.monotonic() - started
                assert elapsed < pg_timeout_s
                assert any("skyrl_clear_lora_receive_target" in n for n in getattr(exc_info.value, "__notes__", []))
                assert "http://server-stalled:8000" in client._dirty_lora_servers()
            else:
                with pytest.raises(PeerRankSyncError, match="Rank 0 failed"):
                    barrier_with_error_propagation(None, stage="in-memory LoRA receive target setup")

            # Server recovers; preflight disarm clears dirty state and round 2 succeeds on the SAME Gloo PG.
            stalled["flag"] = False
            arm_error = None
            if rank == 0:
                await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
            barrier_with_error_propagation(arm_error, stage="in-memory LoRA receive target setup")
            if rank == 0:
                assert armed == {
                    "http://server-stalled:8000": "tenant-b",
                    "http://server-ok:8001": "tenant-b",
                }
                assert client._dirty_lora_servers() == set()

        asyncio.run(_run())
    finally:
        ric_mod.SKYRL_WORKER_NCCL_TIMEOUT_IN_S = orig_pg_timeout
        dist.destroy_process_group()


def test_stalled_arm_and_cleanup_bounded_below_gloo_timeout(tmp_path):
    _run_mp(_worker_stalled_arm_and_cleanup_bounded_below_gloo_timeout, 2, tmp_path)


def _worker_megatron_send_failure_disarms_real_receivers(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _RealStagingReceiver(SkyrlLoraStagingMixin):
            vllm_config = SimpleNamespace(lora_config=object())

        receivers = {
            "http://server-0:8000": _RealStagingReceiver(),
            "http://server-1:8001": _RealStagingReceiver(),
        }

        class _FakeLoraSource(LoraAdapterWeightSource):
            def __init__(self):
                super().__init__(dtype=torch.bfloat16, experts_per_shared_adapter=1)

            def export_adapter_stream(self):
                return [("base_model.model.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.bfloat16))]

        class _FailFirstSendEngine:
            skyrl_force_disable_expandable_segments = False

            def __init__(self, source):
                self.source = source
                self.calls = 0

            def send_weights(self):
                self.calls += 1
                if self.calls == 1:
                    # Simulate rank-0 failing in start_weight_update before finish_weight_update disarms receivers.
                    if rank == 0:
                        assert all(r.skyrl_lora_armed() for r in receivers.values())
                        raise RuntimeError("start_weight_update failed before finish_weight_update")
                    return
                list(self.source.iter_full())
                if rank == 0:
                    # Normal finish_weight_update disarms every receiver.
                    for r in receivers.values():
                        r.skyrl_clear_lora_receive_target()

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="memory")
        source = _FakeLoraSource()
        engine = _FailFirstSendEngine(source)
        worker._lora_weight_source = source
        worker._weight_sync_engine = engine
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)

        client = RemoteInferenceClient.__new__(RemoteInferenceClient)
        client.server_urls = list(receivers.keys())
        loaded = []

        async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
            method_name = json["method"]
            recv = receivers[url]
            if method_name == "skyrl_set_lora_receive_target":
                recv.skyrl_set_lora_receive_target(json["kwargs"]["receive_target"])
                return url, {"status": 200, "body": {}}
            if method_name == "skyrl_clear_lora_receive_target":
                round_id = (json.get("kwargs") or {}).get("round_id")
                recv.skyrl_clear_lora_receive_target(round_id=round_id)
                return url, {"status": 200, "body": {}}
            raise AssertionError(f"Unexpected method {method_name}")

        async def _fake_load_lora(lora_name: str, lora_path=None, *, in_memory: bool = False):
            assert in_memory is True
            loaded.append(lora_name)
            return {}

        client._call_server = _fake_call_server
        client.load_lora_adapter = AsyncMock(side_effect=_fake_load_lora)

        async def _run():
            # Round 1: arm succeeds on real staging receivers, then rank-0 send_weights fails before finish.
            if rank == 0:
                with pytest.raises(RuntimeError, match="start_weight_update failed"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
                assert all(not r.skyrl_lora_armed() for r in receivers.values())
                assert loaded == []
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

            # Round 2: next tenant arms the same real staging receivers without hitting "already armed".
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            if rank == 0:
                assert loaded == ["tenant-b"]
                assert all(not r.skyrl_lora_armed() for r in receivers.values())

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_in_memory_send_failure_disarms_real_staging_receivers(tmp_path):
    _run_mp(_worker_megatron_send_failure_disarms_real_receivers, 2, tmp_path, str(tmp_path / "lora_sync"))


@pytest.mark.asyncio
async def test_failed_disarm_reported_blocks_reuse_until_recovery():
    """A failed disarm preserves the original error with a note and blocks reuse until cleanup succeeds."""
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-a:8000", "http://server-b:8001"]

    armed = {}
    state = {"fail_b_arm": True, "fail_a_clear": True}

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        if method_name == "skyrl_set_lora_receive_target":
            if "server-b" in url and state["fail_b_arm"]:
                raise ConnectionRefusedError("server-b unreachable during arm")
            if url in armed:
                raise RuntimeError(f"already armed for {armed[url]}")
            armed[url] = json["kwargs"]["receive_target"]["lora_name"]
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            if "server-a" in url and state["fail_a_clear"]:
                raise ConnectionError("server-a dropped clear RPC")
            armed.pop(url, None)
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    # Round 1: server-a arms, server-b fails arm, and rollback clear fails on server-a.
    with pytest.raises(ConnectionRefusedError, match="server-b unreachable") as exc_info:
        await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"})
    assert any("server-a:8000" in n for n in getattr(exc_info.value, "__notes__", []))
    assert "http://server-a:8000" in client._dirty_lora_servers()
    assert armed == {"http://server-a:8000": "tenant-a"}

    # Round 2: server-b is healthy now, but server-a's clear still fails -> must reject BEFORE arming server-b.
    state["fail_b_arm"] = False
    with pytest.raises(RuntimeError, match="prior disarm still failing"):
        await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
    assert "http://server-b:8001" not in armed

    # Round 3: server-a recovers -> preflight disarm clears server-a, then both servers arm for tenant-b.
    state["fail_a_clear"] = False
    await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
    assert client._dirty_lora_servers() == set()
    assert armed == {
        "http://server-a:8000": "tenant-b",
        "http://server-b:8001": "tenant-b",
    }


def _worker_blocking_client_stalled_finish_recovers_gloo(rank: int, world_size: int, store_path: str) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        rank0_phase_timeout_s,
    )
    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_timeout,
    )
    from skyrl.backends.skyrl_train.workers.worker_utils import exchange_rank_errors

    pg_timeout_s = 2.5
    _init_gloo_group(rank, world_size, store_path, timeout_s=pg_timeout_s)
    try:
        stalled = {"flag": True}

        class _Resp:
            status_code = 200
            reason = "OK"
            text = "{}"

            def json(self):
                return {"status": "ok"}

        class _BlockingSession:
            def post(self, url, json=None, timeout=None):
                if url.endswith("/finish_weight_update") and stalled["flag"]:
                    # Respect socket timeout if smaller, otherwise stall far past Gloo's 2.5s timeout.
                    time.sleep(min(timeout, 30.0) if timeout is not None else 30.0)
                    raise TimeoutError("simulated read timeout")
                return _Resp()

            def close(self):
                pass

        client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
        client._urls = ["http://server-0:8000"]
        client._dp = 1
        client._init_payload_fn = None
        client._timeout_s = None
        client._start_endpoint = START_UPDATE_ENDPOINT
        client._session = _BlockingSession()
        client._pool = ThreadPoolExecutor(max_workers=1)

        try:
            # Round 1: rank 0 stalls in blocking /finish_weight_update while rank 1 returns early
            # into exchange_rank_errors. Timeout must fire strictly below the 2.5s Gloo PG timeout.
            started = time.monotonic()
            send_error = None
            if rank == 0:
                with weight_sync_client_timeout(rank0_phase_timeout_s(pg_timeout_s)):
                    try:
                        client.finish_weight_update()
                    except Exception as exc:
                        send_error = exc
            any_failed, peer_err = exchange_rank_errors(send_error)
            elapsed = time.monotonic() - started
            assert elapsed < pg_timeout_s
            assert any_failed is True
            if rank == 0:
                assert isinstance(send_error, TimeoutError)
            else:
                assert peer_err is not None and peer_err[0] == 0

            # Round 2: server recovers; same Gloo PG and blocking client both succeed.
            stalled["flag"] = False
            send_error = None
            if rank == 0:
                with weight_sync_client_timeout(rank0_phase_timeout_s(pg_timeout_s)):
                    client.finish_weight_update()
            barrier_with_error_propagation(send_error, stage="blocking finish recovery")
        finally:
            client._pool.shutdown(wait=False)
    finally:
        dist.destroy_process_group()


def test_blocking_weight_sync_client_stalled_finish_recovers_gloo_group(tmp_path):
    _run_mp(_worker_blocking_client_stalled_finish_recovers_gloo, 2, tmp_path)


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_late_server_arm_and_clear_protected_by_round_id():
    """Delayed server-side clear/arm tasks that continue after client timeout cannot corrupt later rounds."""
    from skyrl.backends.skyrl_train.weight_sync.lora_target import (
        build_lora_receive_target,
    )
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    class _RealReceiver(SkyrlLoraStagingMixin):
        vllm_config = SimpleNamespace(lora_config=object())

    receiver = _RealReceiver()
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-0:8000"]

    late_tasks: list[asyncio.Task] = []
    release_late_clear = asyncio.Event()
    release_late_arm = asyncio.Event()
    mode = {"scenario": "delay_clear"}
    late_arm_rejected = {"error": None}

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        kwargs = json.get("kwargs") or {}
        if method_name == "skyrl_set_lora_receive_target":
            target = kwargs["receive_target"]
            if mode["scenario"] == "delay_arm" and target["lora_name"] == "tenant-c":
                # Independent server task that continues even after client times out/cancels.
                async def _server_apply_late_arm():
                    await release_late_arm.wait()
                    try:
                        receiver.skyrl_set_lora_receive_target(target)
                    except Exception as exc:
                        late_arm_rejected["error"] = exc

                late_tasks.append(asyncio.create_task(_server_apply_late_arm()))
                await asyncio.sleep(30.0)
            receiver.skyrl_set_lora_receive_target(target)
            return url, {"status": 200, "body": {}}

        if method_name == "skyrl_clear_lora_receive_target":
            rid = kwargs.get("round_id")
            if mode["scenario"] == "delay_clear":
                mode["scenario"] = "normal"

                # Independent server task that applies the stale clear AFTER tenant-b arms.
                async def _server_apply_late_clear():
                    await release_late_clear.wait()
                    receiver.skyrl_clear_lora_receive_target(round_id=rid)

                late_tasks.append(asyncio.create_task(_server_apply_late_clear()))
                await asyncio.sleep(30.0)
            receiver.skyrl_clear_lora_receive_target(round_id=rid)
            return url, {"status": 200, "body": {}}

        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    # Scenario A: tenant-a's clear times out on client while continuing independently on the server.
    # Preflight clear + tenant-b arm run first, then tenant-a's stale clear arrives on the server.
    await client.set_lora_receive_target(build_lora_receive_target("tenant-a", {}, {}), timeout_s=1.0)
    assert receiver.skyrl_lora_armed() is True
    failed = await client.clear_lora_receive_target(timeout_s=0.15)
    assert "http://server-0:8000" in failed

    # Preflight clears tenant-a cleanly, then arms tenant-b.
    await client.set_lora_receive_target(build_lora_receive_target("tenant-b", {}, {}), timeout_s=1.0)
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-b"

    # Now release the stale clear from tenant-a's round: it must NOT disarm tenant-b.
    release_late_clear.set()
    await asyncio.gather(*late_tasks)
    late_tasks.clear()
    assert receiver.skyrl_lora_armed() is True
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-b"

    # Disarm tenant-b cleanly before Scenario B.
    assert await client.clear_lora_receive_target(timeout_s=1.0) == {}
    assert receiver.skyrl_lora_armed() is False

    # Scenario B: tenant-c's arm times out on client while continuing independently on the server;
    # rollback clear lands on server first, then the delayed arm arrives and must be rejected.
    mode["scenario"] = "delay_arm"
    with pytest.raises(InferenceServerTimeoutError):
        await client.set_lora_receive_target(build_lora_receive_target("tenant-c", {}, {}), timeout_s=0.25)

    release_late_arm.set()
    await asyncio.gather(*late_tasks)
    late_tasks.clear()
    assert isinstance(late_arm_rejected["error"], RuntimeError)
    assert "already cancelled" in str(late_arm_rejected["error"])
    assert receiver.skyrl_lora_armed() is False

    # Tenant-d can now arm cleanly.
    mode["scenario"] = "normal"
    await client.set_lora_receive_target(build_lora_receive_target("tenant-d", {}, {}), timeout_s=1.0)
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-d"


def _worker_megatron_export_copy_failure_drains_collectives(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    from megatron.bridge.models.conversion import peft_bridge

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    orig_build_cfg = peft_bridge.build_adapter_config_dict
    try:
        torch.cuda.empty_cache = lambda: None
        peft_bridge.build_adapter_config_dict = lambda lora_cls, target_modules, base_model_name_or_path: {
            "r": lora_cls.dim,
            "lora_alpha": lora_cls.alpha,
            "target_modules": list(target_modules),
            "base_model_name_or_path": base_model_name_or_path,
        }

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="disk")
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)
        fail_first = {"flag": True}
        seen_cpu_args = []
        second_collective_reached = {"count": 0}

        class _FailingCopyTensor:
            def detach(self):
                return self

            def to(self, *args, **kwargs):
                raise MemoryError("simulated host OOM during writer-only CPU copy")

        def _export_with_collective(actor_module, cpu=True, show_progress=False):
            seen_cpu_args.append(cpu)
            if fail_first["flag"] and rank == 0:
                yield ("layers.0.self_attn.q_proj.lora_A.weight", _FailingCopyTensor())
            else:
                yield ("layers.0.self_attn.q_proj.lora_A.weight", torch.ones(4, 8, dtype=torch.bfloat16))
            # A second collective remains in the stream after the first tensor fails on rank 0.
            dist.barrier()
            second_collective_reached["count"] += 1
            yield ("layers.0.self_attn.q_proj.lora_B.weight", torch.ones(8, 4, dtype=torch.bfloat16))

        worker.bridge.export_adapter_weights = _export_with_collective

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock()

        async def _run():
            # Round 1: writer rank hits MemoryError copying tensor 1 to CPU, must still drain
            # tensor 2's collective barrier without deadlocking peer rank 1.
            if rank == 0:
                with pytest.raises(MemoryError, match="host OOM"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")
                client.load_lora_adapter.assert_not_awaited()
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")

            assert second_collective_reached["count"] == 1
            assert seen_cpu_args == [False]

            # Round 2: export succeeds on both ranks on the same Gloo PG.
            fail_first["flag"] = False
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_2")
            assert second_collective_reached["count"] == 2
            if rank == 0:
                client.load_lora_adapter.assert_awaited_once()

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        peft_bridge.build_adapter_config_dict = orig_build_cfg
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_disk_export_copy_failure_drains_remaining_collectives(tmp_path):
    _run_mp(_worker_megatron_export_copy_failure_drains_collectives, 2, tmp_path, str(tmp_path / "lora_sync"))


def test_blocking_client_deadline_propagates_to_nested_executor():
    """Nested ThreadPoolExecutor submissions (like NCCL sender update_weights) must inherit the deadline."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )

    release = threading.Event()
    seen_timeouts = []

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _HangingSession:
        def post(self, url, json=None, timeout=None):
            seen_timeouts.append(timeout)
            release.wait(timeout=5.0)
            return _Resp()

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _HangingSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    nested_pool = ThreadPoolExecutor(max_workers=1)
    try:
        started = time.monotonic()
        with weight_sync_client_deadline(started + 0.25):
            # Submit from an independent thread pool that does NOT copy ContextVars.
            fut = nested_pool.submit(client.update_weights, {"names": ["w"]})
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                fut.result(timeout=2.0)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert seen_timeouts and seen_timeouts[0] is not None and seen_timeouts[0] <= 0.3
    finally:
        release.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        nested_pool.shutdown(wait=True)
        client._pool.shutdown(wait=True)


def _worker_shared_deadline_bounds_combined_sequential_rpcs(rank: int, world_size: int, store_path: str) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        rank0_phase_timeout_s,
    )
    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )
    from skyrl.backends.skyrl_train.workers.worker_utils import exchange_rank_errors

    pg_timeout_s = 2.5
    phase_budget = rank0_phase_timeout_s(pg_timeout_s)  # 1.5s
    _init_gloo_group(rank, world_size, store_path, timeout_s=pg_timeout_s)
    try:
        slow = {"flag": True}

        class _Resp:
            status_code = 200
            reason = "OK"
            text = "{}"

        class _SlowSequentialSession:
            def post(self, url, json=None, timeout=None):
                if slow["flag"]:
                    # Each RPC takes 1.0s (< 1.5s budget alone), so start + finish = 2.0s > 1.5s.
                    # Respect socket timeout when shorter so the thread settles cleanly.
                    sleep_s = 1.0 if timeout is None else min(1.0, timeout)
                    time.sleep(sleep_s)
                    if timeout is not None and timeout < 1.0:
                        raise TimeoutError("socket timed out")
                return _Resp()

        client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
        client._urls = ["http://server-0:8000"]
        client._dp = 1
        client._init_payload_fn = None
        client._timeout_s = None
        client._start_endpoint = START_UPDATE_ENDPOINT
        client._session = _SlowSequentialSession()
        client._pool = ThreadPoolExecutor(max_workers=1)

        try:
            started = time.monotonic()
            send_error = None
            if rank == 0:
                with weight_sync_client_deadline(started + phase_budget):
                    try:
                        client.start_weight_update()
                        client.finish_weight_update()
                    except Exception as exc:
                        send_error = exc
            any_failed, peer_err = exchange_rank_errors(send_error)
            elapsed = time.monotonic() - started
            assert elapsed < pg_timeout_s
            assert any_failed is True
            if rank == 0:
                assert isinstance(send_error, TimeoutError)
            else:
                assert peer_err is not None and peer_err[0] == 0

            # Drain any finishing thread, recover, and verify the same Gloo PG succeeds on round 2.
            if rank == 0:
                client.drain_inflight(deadline_at=time.monotonic() + 2.0)
            slow["flag"] = False
            if rank == 0:
                with weight_sync_client_deadline(time.monotonic() + phase_budget):
                    client.start_weight_update()
                    client.finish_weight_update()
            barrier_with_error_propagation(None, stage="shared deadline recovery")
        finally:
            client._pool.shutdown(wait=True)
    finally:
        dist.destroy_process_group()


def test_shared_deadline_bounds_combined_sequential_rpcs_on_gloo(tmp_path):
    _run_mp(_worker_shared_deadline_bounds_combined_sequential_rpcs, 2, tmp_path)


def test_timed_out_blocking_rpc_blocks_reuse_until_settled():
    """A blocking POST that ignores socket timeout stays tracked and blocks reuse until it exits."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )

    release_old_post = threading.Event()
    post_count = {"n": 0}

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _StickyBlockingSession:
        def post(self, url, json=None, timeout=None):
            post_count["n"] += 1
            if post_count["n"] == 1:
                # Ignores timeout (simulates a blocking request outliving client timeout).
                release_old_post.wait(timeout=5.0)
            return _Resp()

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _StickyBlockingSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    try:
        # Round 1 times out while the blocking POST thread is still running.
        with weight_sync_client_deadline(time.monotonic() + 0.15):
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                client.finish_weight_update()
        assert post_count["n"] == 1

        # Round 2 while Round 1's thread is still running MUST fail without issuing a second POST.
        with weight_sync_client_deadline(time.monotonic() + 0.15):
            with pytest.raises(TimeoutError, match="prior timed-out request\\(s\\) are still in flight"):
                client.start_weight_update()
        assert post_count["n"] == 1

        # Release the stuck POST thread -> Round 3 drains it cleanly and succeeds.
        release_old_post.set()
        with weight_sync_client_deadline(time.monotonic() + 2.0):
            client.start_weight_update()
        assert post_count["n"] == 2
    finally:
        release_old_post.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        client._pool.shutdown(wait=True)
