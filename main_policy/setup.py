from setuptools import find_packages, setup

package_name = 'main_policy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hungyu',
    maintainer_email='doubleradiate@gmail.com',
    description='Semantic map integrator node',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'map_integrator_node = main_policy.map_integrator_node:main',
        ],
    },
)
