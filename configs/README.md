# Source configurations

Each TOML file pins an upstream revision, names the source-specific importer, lists
the immutable artifacts it consumes, and supplies conservative scenario-provenance
defaults. An artifact without a SHA-256 digest is still revision-pinned; `fetch`
reports its computed digest, which should be recorded before a release is frozen.

These files describe acquisition and parsing. They do not grant authority to the
downloaded content and do not turn upstream labels into PromptSec-FM ground truth.

