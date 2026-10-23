from glob import glob

from setuptools import find_packages, setup

package_name = 'b2_nav_search'

setup(
    name=package_name,
    version='0.1.0',
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
    description='End-stage device search, identity confirmation and observation pose.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'object_search_node = b2_nav_search.object_search_node:main',
        ],
    },
)
