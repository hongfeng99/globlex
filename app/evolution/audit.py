import random


AUDIT_SAMPLE_RATE = 0.05


def select_audit_samples(
    training_batch: list[dict],
) -> list[dict]:
    if not training_batch:
        return []
    count = max(
        1,
        int(
            len(training_batch)
            * AUDIT_SAMPLE_RATE
        ),
    )
    return random.sample(
        training_batch,
        min(count, len(training_batch)),
    )
