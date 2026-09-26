---
name: My repository will not set up
about: eval-harness init, validate or collect does not work on your repository
labels: setup
---

Most setup problems are one of three things: an install command that needs something the base
image lacks, test globs that match nothing, or a base commit whose lockfile differs from today's.

**Please run `eval-harness doctor` and paste the output.** It checks Docker, the image, the
repository path, provider auth and the data root, and names what is wrong.

### What happened

<!-- the command, and what it printed -->

### Your repos.yaml

<!-- the entry for the repository, minus anything private -->

```yaml
```

### The repository

- Language and test runner:
- Public repository, if it is one:
