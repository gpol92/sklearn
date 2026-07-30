import argparse
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description="Configuration of a scikit learn model")
parser.add_argument('-lg', '--logistic', action='store_true', help="Choose this for logistic regression model")
parser.add_argument('rs', type=int, help="Random state of the logistic regression model")
parser.add_argument('max_iter', type=int, help="Maximum number of iterations")
parser.add_argument("solver", choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], help="Choose the solver")
parser.add_argument("--dual", action='store_true', help="Choose to activate dual")

args = parser.parse_args()

if args.logistic:
    print("Logistic regression chosen")
    X, y = load_breast_cancer(return_X_y=True)

    # Rimosso il controllo stringente: argparse valida già i parametri posizionali e i choices

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.rs)

    print(f"Training Logistic Regression with rs={args.rs}, max_iter={args.max_iter}, solver={args.solver}, dual={args.dual}")

    try:
        clf = LogisticRegression(max_iter=args.max_iter, random_state=0, solver=args.solver, dual=args.dual)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test)) * 100
        print(f"Logistic Regression model accuracy: {acc:.2f}%")

    except ValueError as e:
        print(f"\n[ERRORE SCIKIT-LEARN] Combinazione di parametri non valida: {e}")

else:
    print("Modello logistico non selezionato. Usa il flag -lg o --logistic.")
