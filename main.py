import contract_dz
from flask_restful import Resource
class Todo(Resource):
    def get(self, id):
        question = "What is the name of the Agreement Title?"
        out_d = contract_dz.contract_digitization(question)
        return out_d

