import typer

from anaconda_cloud_auth.client import BaseClient
from anaconda_cloud_auth.console import console

app = typer.Typer(add_completion=False, help="Anaconda.cloud auth commands")


@app.command(name="info")
def auth_info() -> None:
    """Display information about the currently signed-in user"""
    client = BaseClient()
    response = client.get("/api/account")
    console.print("Your Anaconda Cloud info:")
    console.print(response.json())
