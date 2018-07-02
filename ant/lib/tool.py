import os, sys

def double_check(*args, **kwargs):
    print("\n# CheckList : " )
    print(args[0].keys()[0])
    for i, kwarg in enumerate(kwargs):
        print("\n** {} is : {} ".format(kwarg, kwargs[kwarg]))

    check = input("\n# Y/N ? :").lower()
    if check in ["y", "yes"] :
        print("\n# Pass")
        pass
    elif check in ("n", "no", "false"):
        print("\n# Exit")
        return sys.exit()
