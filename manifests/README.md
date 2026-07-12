# Source and license manifests

The manifests record component-level upstream licensing and redistribution caveats.
They are evidence for release review, not legal advice. A source-level software
license must never be assumed to cover bundled or dynamically loaded datasets.

Every release must re-check upstream terms, retain required attribution, and exclude
components whose redistribution rights are absent or unclear.

`source-lock.json` mirrors the immutable revisions, exact package pins, and expected
artifact SHA-256 values from the six tracked source configurations. It is a compact
reproducibility index, not a second license manifest and not evidence that a payload
may be redistributed. Source TOML files remain the acquisition authority; tests
require the lock and configurations to agree.
