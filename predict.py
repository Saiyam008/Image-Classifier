import json


from predict_functions import get_input_args
from predict_functions import process_image
from predict_functions import load_checkpoint_file
from predict_functions import predict
from predict_functions import plotting


def main():

    in_arg = get_input_args()
    print('Arguments in')


    img = process_image(in_arg.image_path)
    print('Image processed')


    model = load_checkpoint_file(in_arg.checkpoint_file, in_arg.arch)
    print('Rebuilt model and reloaded model state')

    ps_topk, cat_topk = predict(img, model, in_arg.topk)
    print('Prediction done')

    with open(in_arg.category_names, 'r') as f:
        cat_to_name_dict = json.load(f)


    labels = [cat_to_name_dict[clas] for clas in cat_topk]
    ps_topk_formatted = ["%.1f" % (ps * 100) for ps in ps_topk]


    print('Top {} predicted categories and probabilities'.format(in_arg.topk))
    print('    Categories: {}'.format(labels))
    print('    Probabilities, %: ', ps_topk_formatted)

    print('END')


if __name__ == "__main__":
    main()