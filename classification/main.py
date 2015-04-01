__author__ = 'piyush'

from BankNoteAuth import bankNote
from BreastCancer import breastCancer
from ClimateModel import climateModel
from Ecoli import ecoli
from OptDigits import digits

def main():
    print('Learning for Bank Note Auth ......')
    bankNote()
    print

    print('Learning for Breast Cancer ......')
    breastCancer()
    print

    print('Learning for Climate Model ......')
    climateModel()
    print

    print('Learning for Ecoli ......')
    ecoli()
    print

    print('Learning for Optical Digits ......')
    digits()
    print


if __name__ == '__main__':
    main()