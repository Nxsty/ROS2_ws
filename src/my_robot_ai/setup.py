from setuptools import find_packages, setup

package_name = 'my_robot_ai'

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
    maintainer='mfederi',
    maintainer_email='mauriciofecasillas@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inspect_h5 = my_robot_ai.inspect_h5:main',
            'transformer_autopilot = my_robot_ai.transformer_autopilot:main',
            'advanced_collector = my_robot_ai.advanced_lidar_collector:main',
            'goal_collector = my_robot_ai.goal_autopilot:main',
            'advanced_manual_collector = my_robot_ai.advanced_manual_collector:main',
            'ultimate_collector = my_robot_ai.ultimate_manual_collector:main',
            'where_am_i = my_robot_ai.where_am_i:main',
            'get_pose = my_robot_ai.get_pose:main',
            'circuit_autopilot = my_robot_ai.circuit_autopilot:main',
        ],
    },
)
