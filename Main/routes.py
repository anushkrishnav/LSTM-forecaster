from Main.api import ForecastApi


def initialize_routes(api):
    # List users
    api.add_resource(ForecastApi, '/api/forecast') 