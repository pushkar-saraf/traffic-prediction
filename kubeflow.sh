kubectl delete --all pods --namespace=kube-system
kubectl delete --all pods --namespace=kubeflow
sudo sysctl fs.inotify.max_user_instances=1280
sudo sysctl fs.inotify.max_user_watches=655360
export PIPELINE_VERSION=2.3.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
until kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80; do echo "Failed.. retrying"; sleep 1; done;
