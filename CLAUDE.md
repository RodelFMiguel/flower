# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Flower is a production-ready federated learning framework built with extensibility and understandability as core principles. The framework separates long-running infrastructure (SuperLink/SuperNode) from short-lived training logic (ServerApp/ClientApp), enabling multi-tenancy and concurrent training runs.

**Key Architecture Pattern:**
```
SuperLink (long-running)  ←→  ServerApp (per-run, strategy logic)
SuperNode (long-running)  ←→  ClientApp (per-run, training logic)
```

## Repository Structure

```
/flower
├── framework/              # Core Python framework (main development area)
│   ├── py/flwr/           # Source code
│   ├── proto/             # Protocol Buffer definitions
│   ├── docs/              # Sphinx documentation
│   ├── dev/               # Development scripts
│   └── pyproject.toml     # Poetry config (v1.25.0)
├── examples/              # 30+ integration examples
├── baselines/             # 30+ research baseline implementations
└── datasets/              # Flower Datasets package
```

## Development Workflow

### Initial Setup

```bash
cd framework
pip install poetry==2.1.3
python -m poetry install --all-extras
```

**Python Support:** 3.10, 3.11, 3.12, 3.13

### Common Commands

**Format code before committing:**
```bash
./framework/dev/format.sh
```
This runs: taplo, copyright check, init_py_fix, isort, black, docformatter, ruff --fix, clang-format (protos), mdformat, docstrfmt

**Run all checks and tests:**
```bash
./framework/dev/test.sh
```
This runs: clang-format check, isort check, black check, init_py_check, docformatter check, docsig, ruff, mypy, pylint, pytest with coverage, mdformat check, taplo check, docstrfmt check, copyright check, licensecheck

**Quick test (skip docs/e2e):**
```bash
./framework/dev/test.sh false
```

**Build distribution:**
```bash
./framework/dev/build.sh
```

**Build documentation:**
```bash
./framework/dev/build-docs.sh
```

**Compile Protocol Buffers:**
```bash
python -m flwr_tool.protoc
```

**Run specific test file:**
```bash
cd framework
python -m pytest py/flwr/server/strategy/fedavg_test.py
```

**Run tests with coverage:**
```bash
python -m pytest --cov=py/flwr py/flwr
```

## Code Quality Standards

**Style:**
- Line length: 88 characters (Black default)
- Docstrings: NumPy convention (enforced by ruff)
- Type hints: Strict mypy checking required
- Import organization: isort with Black profile
- All files require Apache 2.0 license headers

**Testing:**
- Test files: `*_test.py` suffix, co-located with source
- Framework: pytest with unittest.mock
- All new code requires tests
- Run `pytest --cov=py/flwr` to verify coverage

**CI Requirements:**
All checks in `test.sh` must pass before merging:
- Code formatting (black, isort, docformatter)
- Type checking (mypy with strict mode)
- Linting (ruff with D, E, F, W, B, ISC, C4, UP rules; pylint)
- Docstring validation (docsig)
- Tests (pytest with coverage)
- License compliance

## Architecture Deep Dive

### Core Packages (framework/py/flwr/)

**Deployment Engine:**
- `superlink/` - Server-side coordination agent (long-running federation manager)
- `supernode/` - Client-side deployment agent (long-running task executor)
- `supercore/` - Core execution engine with object store and file-based storage
- `cli/` - Command-line interface with multiple subcommands

**Application Framework:**
- `serverapp/` - Server-side application framework defining aggregation logic
- `clientapp/` - Client-side application framework defining training/eval logic
- `server/strategy/` - 20+ federated learning algorithms (FedAvg, FedProx, FedAdam, etc.)

**Communication:**
- `proto/` - Auto-generated Protocol Buffer classes (excluded from imports)
- `common/` - Shared utilities: Message, Parameters, MetricRecord, logger, telemetry

**Legacy:**
- `server/` - Low-level server implementation (older API)
- `client/` - Low-level client implementation (NumPyClient, older API)
- `compat/` - Backward compatibility layer

**Optional:**
- `simulation/` - Ray-based simulator for local federated learning

### Strategy Pattern

All FL algorithms inherit from `Strategy` abstract base class:
```python
class Strategy(ABC):
    def configure_train(server_round, arrays, config, grid) -> Iterable[Message]
    def aggregate_train(server_round, replies) -> tuple[ArrayRecord, MetricRecord]
    def configure_evaluate(server_round, arrays, config, grid) -> Iterable[Message]
    def aggregate_evaluate(server_round, replies) -> MetricRecord
    def start(grid, initial_arrays, num_rounds) -> Result
```

**Built-in strategies:** FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, FedAvgM, QFedAvg, FedMedian, Krum, Bulyan, FedTrimmedAvg, FaultTolerantFedAvg, DPFedAvgFixed, DPFedAvgAdaptive, FedXgbBagging, FedXgbCyclic, FedXgbNnAvg

### Message-Oriented Driver (MOD) Pattern

The `mod/` directories implement composable message transformations for features like differential privacy and secure aggregation:
- `comms_mods.py` - Communication layer modifications
- `localdp_mod.py` - Local differential privacy
- `centraldp_mods.py` - Central differential privacy
- `secure_aggregation/` - Cryptographic aggregation protocols

### Context Pattern

Unified context object passed through the system:
```python
context: Context
context.node_id           # Unique node identifier
context.node_config       # Node-specific configuration (e.g., dataset path)
context.run_config        # Run-wide hyperparameters from pyproject.toml
context.server_round      # Current federated learning round
context.state            # Shared state storage
```

## Working with Examples

Examples are self-contained projects with their own `pyproject.toml`:

```bash
cd examples/quickstart-pytorch
pip install -e .
# Follow example's README.md
```

Example structure:
- `<package>/client_app.py` - ClientApp with @app.train() and @app.evaluate()
- `<package>/server_app.py` - ServerApp with strategy initialization
- `<package>/task.py` - Model definition and training/eval functions
- `pyproject.toml` - Dependencies and [tool.flwr.app] configuration

## Protocol Buffers

**Location:** `framework/proto/flwr/proto/*.proto`

**Key files:**
- `federation.proto` - Federation coordination
- `transport.proto` - Message transport
- `node.proto` - Node communication
- `control.proto` - Control flow

**After modifying protos:**
```bash
python -m flwr_tool.protoc
./framework/dev/format.sh
```

## Documentation

**Location:** `framework/docs/source/`

**Build locally:**
```bash
./framework/dev/build-docs.sh
# Output: framework/docs/build/html/index.html
```

**Documentation types:**
- `explanation-*.rst` - Architecture and design decisions
- `contributor-how-to-*.rst` - Developer guides
- `contributor-tutorial-*.rst` - Onboarding tutorials

**Adding new docs:**
1. Create `.rst` file in `framework/docs/source/`
2. Add to appropriate `toctree` in index files
3. Run `./framework/dev/build-docs.sh` to verify
4. Ensure `docstrfmt` passes (included in `format.sh`)

## Testing Philosophy

- Tests co-located with source (`*_test.py` suffix)
- Use `unittest.mock` for mocking dependencies
- Parameterized tests via `parameterized` library
- Test both success and failure paths
- Integration tests in `framework/e2e/`
- Exclude proto files from testing: `py/flwr/proto/*` is auto-generated

## API Stability

**Public APIs:** Stable, follow semantic versioning
**Private APIs:** (prefixed with `_`) may change without notice

See `framework/docs/source/contributor-explanation-public-and-private-apis.rst`

## Common Development Patterns

**Adding a new strategy:**
1. Create `py/flwr/server/strategy/mystrategy.py`
2. Inherit from `Strategy` base class
3. Implement all abstract methods
4. Add to `py/flwr/server/strategy/__init__.py`
5. Create `mystrategy_test.py` with comprehensive tests
6. Update documentation

**Adding a new example:**
1. Create directory in `examples/my-example/`
2. Create `myexample/` package with `client_app.py`, `server_app.py`, `task.py`
3. Add `pyproject.toml` with `[tool.flwr.app]` configuration
4. Add comprehensive `README.md` with setup and run instructions
5. Test with both simulation and deployment modes

**Modifying protos:**
1. Edit `.proto` files in `framework/proto/flwr/proto/`
2. Run `python -m flwr_tool.protoc` to regenerate Python code
3. Update any affected code in `py/flwr/`
4. Run full test suite to ensure compatibility

## CI/CD

**Pre-commit hook (optional but recommended):**
```bash
pre-commit install
```

This runs `format.sh` and `test.sh` on every commit. Bypass with `git commit --no-verify`.

**GitHub Actions:** `.github/workflows/framework.yml`
- All checks from `test.sh` run on every PR
- Can run locally with `act` tool

## Key Dependencies

**Core:**
- numpy >=1.26.0
- grpcio ^1.70.0 - gRPC communication
- protobuf >=5.28.0 - Protocol buffers
- cryptography ^44.0.1 - Security primitives
- typer >=0.12.5 - CLI framework
- rich ^13.5.0 - Terminal output

**Optional:**
- ray ==2.51.1 - Simulation engine (`[simulation]` extra)
- starlette ^0.45.2 - REST transport (`[rest]` extra)

## Useful References

**Main documentation:** https://flower.ai/docs
**Architecture guide:** `framework/docs/source/explanation-flower-architecture.rst`
**Contribution tutorial:** `framework/docs/source/contributor-tutorial-contribute-on-github.rst`
**Strategy abstraction:** `framework/docs/source/explanation-flower-strategy-abstraction.rst`

## Multi-Run and Multi-Tenancy

Flower's Deployment Engine supports running multiple concurrent federated learning jobs (runs) within the same infrastructure:
- Multiple `ServerApp` instances can run simultaneously on one `SuperLink`
- Multiple `ClientApp` instances can run simultaneously on one `SuperNode`
- Each run is isolated with its own configuration and state
- Use `flwr run` to start new runs on existing infrastructure

## CLI Entry Points

Defined in `pyproject.toml`:
- `flwr` - Main CLI interface
- `flower-superlink` - Start SuperLink server
- `flower-supernode` - Start SuperNode client
- `flwr-simulation` - Run simulation mode (requires `[simulation]` extra)
- `flwr-serverapp` - Legacy ServerApp CLI
- `flwr-clientapp` - Legacy ClientApp CLI
