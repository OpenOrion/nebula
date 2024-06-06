from setuptools import setup

setup(
    name="nebula",
    version="1.0.0",
    description="auto differentiable CAD library in JAX",
    author="Afshawn Lotfi",
    author_email="",
    packages=[
        "nebula",
        "nebula.cases",
        "nebula.prim",
        "nebula.evaluators",
        "nebula.helpers",
        "nebula.render",
        "nebula.tools",
        "nebula.topology",
    ],
    install_requires=[
        "plotly==5.11.0",
        "ipywidgets==7.6",
        "jupyterlab",
        "matplotlib",
        "numpy",
        "ipython_genutils",
        "jupyter_cadquery",
        "jax[cpu]",
        "jax_dataclasses",
        "kaleido"
    ],
)
