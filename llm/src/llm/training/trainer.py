import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from llm.training.optimizer import get_optimizer
from llm.training.scheduler import get_linear_schedule_with_warmup


class Trainer:
    """
    Универсальный класс обучения LLM (GPT, LLaMA, Mistral и т.д.)
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        lr=3e-4,
        batch_size=8,
        num_epochs=3,
        warmup_steps=100,
    ):
        self.model = model
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        )
        self.optimizer = get_optimizer(model, lr=lr)
        self.scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

    def compute_lm_loss(self, logits, labels):
        """
        Вычисляет loss для языкового моделирования.
        Сдвигает логиты и метки для предсказания следующего токена.
        """
        # Сдвигаем логиты и метки для языкового моделирования
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Вычисляем cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # Игнорируем padding tokens
        )
        return loss

    def train(self):
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, self.warmup_steps, total_steps
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            for batch in progress_bar:
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Универсально обрабатываем выход (tuple/logits)
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Trainer вычисляет loss
                loss = self.compute_lm_loss(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} finished — avg loss: {avg_loss:.4f}")

            if self.val_loader:
                self.evaluate()

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = self.compute_lm_loss(logits, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation loss: {avg_loss:.4f}")
