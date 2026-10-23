from glob import glob

from setuptools import find_packages, setup

package_name = 'b2_nav_experiment'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config/experiments',
            glob('config/experiments/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dministrator',
    maintainer_email='dministrator@todo.todo',
    description='Experiment recording, replay, failure analysis and dataset export.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'experiment = b2_nav_experiment.cli:main',
        ],
    },
)
