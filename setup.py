from setuptools import find_packages, setup

package_name = 'state_space_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/examples',
            ['examples/lqr_design.yaml', 'examples/hinf_design.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rahgirrafi',
    maintainer_email='rahgirrafi@gmail.com',
    description='Modular controller synthesis (LQR, LQG, H-infinity, ...) '
                'for linear state-space plants.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ss_design = state_space_control.cli:main',
        ],
    },
)
