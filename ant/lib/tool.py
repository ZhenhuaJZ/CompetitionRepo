import os, sys

def double_check(dict, **kwargs):
    print("\n# CheckList : " )
    
    for i in dict:
        print("\n** {} is : {} ".format(i, dict[i]))
    for i, kwarg in enumerate(kwargs):
        print("\n** {} is : {} ".format(kwarg, kwargs[kwarg]))

    check = input("\n# Y/N ? :").lower()
    if check in ["y", "yes"] :
        print("\n# Pass")
        pass
    elif check in ("n", "no", "false"):
        print("\n# Exit")
        return sys.exit()
