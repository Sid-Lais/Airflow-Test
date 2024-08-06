from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(ti):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    ti.xcom_push(key='train_test_data', value=(X_train, X_test, y_train, y_test))

def train_model(ti):
    X_train, X_test, y_train, y_test = ti.xcom_pull(key='train_test_data', task_ids='load_data')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ti.xcom_push(key='model', value=model)

def evaluate_model(ti):
    model = ti.xcom_pull(key='model', task_ids='train_model')
    X_train, X_test, y_train, y_test = ti.xcom_pull(key='train_test_data', task_ids='load_data')
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")

with DAG("ml_workflow_dag",
         start_date=datetime(2023, 1, 1),
         schedule_interval=None,
         catchup=False) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    load_data_task >> train_model_task >> evaluate_model_task
