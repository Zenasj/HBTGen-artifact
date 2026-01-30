py
if actual.is_mps or expected.is_mps:
    actual = actual.cpu()
    expected = expected.cpu()

if actual.is_mps or expected.is_mps:  # type: ignore[attr-defined]
            actual = actual.cpu()
            expected = expected.cpu()