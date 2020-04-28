from billTemplate.resources.inference import Inference,InferenceHealth

def registry_resource(api):
    api.add_resource(Inference, "/api/v1alpha1/inference")
    api.add_resource(InferenceHealth,'/api/v1alpha1/status')
