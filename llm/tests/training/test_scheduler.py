import torch
import torch.nn as nn
from llm.training.scheduler import get_linear_schedule_with_warmup
from llm.training.optimizer import get_optimizer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

def test_scheduler_warmup_and_decay():
    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 5
    total_steps = 20
    optimizer = get_optimizer(model, lr=base_lr, optimizer_type="sgd")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    lrs = [optimizer.param_groups[0]['lr']]  # lr до первого .step()
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Проверяем warmup: lr должен расти линейно в первых warmup_steps (начиная с шага 1)
    for i in range(warmup_steps + 1):
        expected = base_lr * min(i, warmup_steps) / max(1, warmup_steps)
        assert abs(lrs[i] - expected) < 1e-6, f"Warmup step {i}: lr={lrs[i]}, expected={expected}"
    # Проверяем decay: после warmup lr затухает
    for i in range(warmup_steps + 1, total_steps + 1):
        expected = base_lr * max(0.0, (total_steps - (i - 0)) / max(1, total_steps - warmup_steps))
        assert abs(lrs[i] - expected) < 1e-6, f"Decay step {i}: lr={lrs[i]}, expected={expected}"
    assert lrs[-1] == 0.0

def test_scheduler_no_warmup():
    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 0
    total_steps = 10
    optimizer = get_optimizer(model, lr=base_lr, optimizer_type="adam")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    lrs = [optimizer.param_groups[0]['lr']]
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    for i in range(total_steps + 1):
        expected = base_lr * max(0.0, (total_steps - i) / max(1, total_steps - warmup_steps))
        assert abs(lrs[i] - expected) < 1e-6, f"Step {i}: lr={lrs[i]}, expected={expected}"
    assert lrs[-1] == 0.0

def test_scheduler_full_decay_to_zero():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=1.0, optimizer_type="adamw")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=2)
    scheduler.step()
    scheduler.step()
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 0.0
