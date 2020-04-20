from setuptools import setup, find_packages

setup(name='MF_Sim',
      version='0.1.0',
      description='High-Fidelity and Low-Fidelity Simulator',
      url='not update',
      author='Jiantao Qiu & Weiling Liu',
      author_email='qjt15@mails.tsinghua.edu.cn',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym',
                        'numpy-stl',
                        'six',
                        'pyglet']
)
