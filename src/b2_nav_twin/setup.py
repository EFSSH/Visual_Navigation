from glob import glob

from setuptools import find_packages, setup

package_name = 'b2_nav_twin'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dministrator',
    maintainer_email='dministrator@todo.todo',
    description='Local geometric world model for B2 navigation.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'local_world_model_node = '
            'b2_nav_twin.local_world_model_node:main',
        ],
    },
)
