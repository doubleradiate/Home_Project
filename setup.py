from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vision_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='work',
    maintainer_email='work@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "grounding_dino_node = vision_package.grounding_dino_node:main",
            "ram_node = vision_package.ram_node:main",
            "bbox_center_3d_node = vision_package.bbox_center_3d_node:main",
            "bbox_all_3d_marker_node = vision_package.bbox_all_3d_marker_node:main",
        ],
    },
)