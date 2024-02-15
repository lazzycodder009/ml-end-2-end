from setuptools import find_packages,setup
from typing import List



HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements =[]
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [ x.replace('\n','') for x in requirement]
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
        requirements.append(requirement)
    
    return requirements


if __name__== '__main__':
    print(get_requirements('requirements.txt'))
