### Installation notes
It is preferable to work in an environment to have the used packages organized. Thus:

```shell
pip install virtualenv
virtualenv py_env
source py_env/bin/activate
```


Then, install the project requirements
```
pip install -r requirements.txt
```

To deactivate the environment, just run `deactivate`

To add a new package to the requirements file, run `pip freeze > requirements.txt`