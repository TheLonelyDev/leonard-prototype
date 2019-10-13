# ModelBuilder
# This script is just made to create A & V models once and save them using a dataset
from src.modules.ModelFactory import ModelFactory

mf = ModelFactory('C:\\Users\\Lonely\\PycharmProjects\\leonard-prototype\\src\\data', 'data4_min.json')

#

mf.CreateArousalModel()
mf.CreateValenceModel()