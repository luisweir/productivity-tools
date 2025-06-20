#!/usr/bin/env python3
# load-config.py - Module to load configuration properties from oci.env for OCI and LangChain scripts.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - oci.env file in the same directory with OCI GenAI and LangChain settings
#
# Usage:
#   from load_config import LoadConfig
#   props = LoadConfig()
class LoadConfig:

    def __init__(self):
        import json
        with open('oci.env') as f:
            data = f.read()
        js = json.loads(data)

        self.default_profile = js.get("default_profile")
        self.model_name = js.get("model_name")
        self.model_ocid = js.get("model_ocid")
        self.endpoint = js.get("endpoint")
        self.compartment_ocid = js.get("compartment_ocid")
        self.embedding_model_name = js.get("embedding_model_name")
        self.langchain_key = js.get("langchain_key")
        self.langchain_endpoint = js.get("langchain_endpoint")

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

    def getLangChainEndpoint(self):
        return self.langchain_endpoint