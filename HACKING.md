Modules:

- activation maximisation (activemax)
- layer activation
- CONV/FC filter visualisation (filtervis)


Ideas:

Extend models by adding a custom bound method, rather than passing it to
an object of the custom class?

```python

from types import MethodType

model = models.alexnet(pretrained=True)

def custom_func(self):
    print('bound custom func!')

model.custom_func = MethodType(custom_func, model)

model.custom_func()  # prints out 'bound custon func!'
```

TODO:

- add test script to travis config
