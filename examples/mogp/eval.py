from gpplus.eval.evaluator import Evaluator

from model import FileldModel
from data import generate_data
from plot import plot_2d

def main():

    # Generate data
    train_x, train_y = generate_data(nx = 30, ny = 30, nsamples = 2)

    # Model
    model = FileldModel(
        x_train = train_x,
        y_train = train_y
    )

    model.load()

    # Train
    evaluator = Evaluator(
        model = model,
    )

    plot_2d(train_x, train_y, evaluator, 30, 30, 0)

if __name__ == "__main__":
    main()