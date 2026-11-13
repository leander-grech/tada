# Simulator wheels

Flight simulator 0.2.80 builds for CPython 3.12 are shared by both implementations.

| Environment | Wheel in `wheels/` |
|---|---|
| Linux x86_64, glibc 2.34 or newer | `flight_simulator-0.2.80-cp312-cp312-manylinux_2_34_x86_64.whl` |
| macOS 11 or newer, Apple Silicon | `flight_simulator-0.2.80-cp312-cp312-macosx_11_0_arm64.whl` |
| Windows x86_64 | `flight_simulator-0.2.80-cp312-cp312-win_amd64.whl` |

Copy the wheel matching your Python environment into `multi_agent/simulator/wheels/` or `single-agent/simulator/wheels/`, depending on the implementation you use; create the destination directory if needed. Keep the filename unchanged. Use the Linux wheel inside WSL or a Linux container.

Install the copied wheel in the same Python environment as the implementation. Multi-agent setup accepts an explicit wheel with `--native-wheel simulator/wheels/<filename>`; single-agent installation uses `python -m pip install simulator/wheels/<filename>`.

The implementation guides document Linux setup. Availability of a simulator wheel does not establish compatibility of the full training stack on other operating systems.

Verify the shared files from the repository root:

```sh
(cd artifacts/simulator && sha256sum -c SHA256SUMS)
```
