# FrameTrajectory

FrameTrajectory allows for storing SE3 elements in a time ordered fashion. It takes inspiration from Molecular Dynamics analysis libraries such as MDAnalysis, simpletraj, ProDy, etc, but instead of 3 dimensional Euclidian space positions it stores coordinate frames (SE3). It is for example used to represent coarse grained atom groups from MD simulations.

## Dependencies

- Python 3.6 and up.
- numpy 1.19
