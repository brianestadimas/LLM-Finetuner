from tqdm import tqdm
import sys
import time

def run_training(num_epochs=5):
    # Force tqdm to show the bar
    for epoch in tqdm(
        range(1, num_epochs + 1),
        desc="Training Progress",
        unit="epoch",
        file=sys.stdout,
        ncols=80,
        ascii=False,
        dynamic_ncols=False,
        leave=True,
        force=True   # <-- Force display even if not in a TTY
    ):
        time.sleep(1)  # Simulate training
        print(f"Completed epoch {epoch}/{num_epochs}")
        sys.stdout.flush()

if __name__ == "__main__":
    run_training(5)
