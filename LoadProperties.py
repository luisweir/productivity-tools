# LoadProperties.py - Module to load configuration properties from config.txt for OCI and LangChain scripts.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - config.txt file in the same directory with keys: model_name, embedding_model_name, endpoint, compartment_ocid, langchain_key, langchain_endpoint
#
# Usage:
#   from LoadProperties import LoadProperties
#   props = LoadProperties()
class LoadProperties:

    def __init__(self):

        import json
        # reading the data from the file
        with open('oci.env') as f:
            data = f.read()

        js = json.loads(data)

        self.default_profile = js["default_profile"]
        self.model_name = js["model_name"]
        self.model_ocid = js["model_ocid"]
        self.endpoint = js["endpoint"]
        self.compartment_ocid = js["compartment_ocid"]
        self.embedding_model_name=js["embedding_model_name"]
        self.langchain_key = js["langchain_key"]
        self.langchain_endpoint = js["langchain_endpoint"]

    def getDefaultProfile(self):
            return self.default_profile
        
    def getModelName(self):
            return self.model_name
    
    def getModelOcid(self):
            return self.model_ocid
    
    def getEndpoint(self):
            return self.endpoint

    def getCompartment(self):
            return self.compartment_ocid

    def getEmbeddingModelName(self):
            return self.embedding_model_name

    def getLangChainKey(self):
            return self.langchain_key

    def getlangChainEndpoint(self):
            return self.langchain_endpoint