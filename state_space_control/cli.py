"""Design a controller for a saved plant from a YAML spec::

    ss_design plant.npz design.yaml [-o controller.npz]

where plant.npz comes from urdf_state_space (StateSpaceModel.save_npz or the
urdf2ss export) and design.yaml looks like::

    controller: lqr
    params:
      Q: [100, 100, 1, 1]     # scalar, diagonal list, or full matrix
      R: 0.1
"""

import argparse
import sys

import numpy as np
import yaml

from .base import Plant, available_controllers, make_controller


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='ss_design',
        description='Synthesize a controller for a state-space plant. '
                    f'Available controllers: {available_controllers()}')
    parser.add_argument('plant', help='.npz plant from urdf_state_space')
    parser.add_argument('design', help='YAML design spec')
    parser.add_argument('-o', '--out', default=None,
                        help='save the result to a .npz file')
    args = parser.parse_args(argv)

    with open(args.design) as f:
        spec = yaml.safe_load(f) or {}
    if 'controller' not in spec:
        parser.error(f'{args.design}: missing required key "controller"')

    plant = Plant.from_npz(args.plant)
    design = make_controller(spec['controller'], **(spec.get('params') or {}))
    result = design.design(plant)

    np.set_printoptions(precision=5, suppress=True, linewidth=120)
    print(result.summary())

    if args.out:
        result.save_npz(args.out)
        print(f'\nsaved {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
