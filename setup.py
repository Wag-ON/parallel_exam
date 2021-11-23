import os
import setuptools

setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, 'requirements.txt'), 'r') as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

setuptools.setup(
      name='parallel_exam',
      install_requires=requirements
)