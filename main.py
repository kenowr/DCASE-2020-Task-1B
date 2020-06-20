import argparse

parser = argparse.ArgumentParser(description='Trains and outputs metrics for the models specified in the DCASE 2020 Task 1B submission.')
parser.add_argument(dest='model_nos', metavar='model_nos', type = int, nargs='*', default = 0, choices = [0,1,2,3,4],
                    help='The model numbers to train. If 0 is specified, all models (1-4) are trained. Briefly, Model 1 is a VGGNet-based model, Model 2 is an InceptionNet-based model, Model 3 is an ensemble of 5 VGGNet-based and 1 InceptionNet-based models trained on non-augmented data, and Model 4 has the same network architecutre as Model 3 but is trained on augmented data. Please see the technical report for more details.')

args = parser.parse_args()

if isinstance(args.model_nos,int):
    args.model_nos = [args.model_nos]

for model_no in args.model_nos:
    if model_no == 0:
        import model1
        import model2
        import model3
        import model4
    elif model_no == 1:
        import model1
    elif model_no == 2:
        import model2
    elif model_no == 3:
        import model3
    else:
        import model4