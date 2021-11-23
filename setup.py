import os
import setuptools

setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, 'requirements.txt'), 'r') as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

setuptools.setup(
      name='parallel_exam',
      # version=get_version('__init__.py'),
      # description='Betting tool',
      # include_package_data=True,
      # packages=setuptools.find_packages(),
      install_requires=requirements
)