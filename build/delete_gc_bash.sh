#!/bin/bash

# Copyright (c) 2025. All rights reserved.
# DS 5460: Big Data Scaling
# Spring 2025
# Author: Yehyun Suh; yehyun.suh@vanderbilt.edu
# Description: This script automates the deletion of Google Cloud resources, including projects, buckets, and clusters.
# License: This script is provided "as-is" without warranty of any kind. Use at your own risk.

# Example Usage
# This script allows you to delete Google Cloud resources using the following commands:
#
# 1. Delete all resources (project, bucket, and cluster):
#    bash delete_gc_bash.sh -p_id PROJECT_ID -b_n BUCKET_NAME -c_n CLUSTER_NAME -region REGION
#    Example: bash delete_gc_bash.sh -p_id my-project -b_n my-bucket -c_n my-cluster -region us-central1
#
# 2. Delete only a project:
#    bash delete_gc_bash.sh -p -p_id PROJECT_ID
#    Example: bash delete_gc_bash.sh -p -p_id my-project
#
# 3. Delete only a bucket:
#    bash delete_gc_bash.sh -b -p_id PROJECT_ID -b_n BUCKET_NAME
#    Example: bash delete_gc_bash.sh -b -p_id my-project -b_n my-bucket
#
# 4. Delete only a cluster:
#    bash delete_gc_bash.sh -c -p_id PROJECT_ID -b_n BUCKET_NAME -c_n CLUSTER_NAME -region REGION
#    Example: bash delete_gc_bash.sh -c -p_id my-project -b_n my-bucket -c_n my-cluster -region us-central1
#
# 5. Delete a bucket and cluster:
#    bash delete_gc_bash.sh -b -c -p_id PROJECT_ID -b_n BUCKET_NAME -c_n CLUSTER_NAME -region REGION
#    Example: bash delete_gc_bash.sh -bc -p_id my-project -b_n my-bucket -c_n my-cluster -region us-central1
#
# Notes:
# - PROJECT_ID is mandatory for all operations.
# - BUCKET_NAME is required for operations involving buckets or clusters.
# - CLUSTER_NAME is required for operations involving clusters.
# - REGION is required for operations involving clusters.

# Exit on any error
set -e

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS] -p_id PROJECT_ID [-b_n BUCKET_NAME] [-c_n CLUSTER_NAME] [-region REGION]"
    echo "Options:"
    echo "  -p             Delete project only"
    echo "  -b             Delete bucket only"
    echo "  -c             Delete cluster only"
    echo "  -bc            Delete bucket and cluster"
    echo "  -pbc           Delete project, bucket, and cluster"
    echo "No options:      Delete project, bucket, and cluster"
    exit 1
}

# Default flags
DELETE_PROJECT=false
DELETE_BUCKET=false
DELETE_CLUSTER=false
REGION="us-central1"

# Parse options
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -p)
      DELETE_PROJECT=true
      shift
      ;;
    -b)
      DELETE_BUCKET=true
      shift
      ;;
    -c)
      DELETE_CLUSTER=true
      shift
      ;;
    -bc)
      DELETE_BUCKET=true
      DELETE_CLUSTER=true
      shift
      ;;
    -p_id)
      PROJECT_ID="$2"
      shift 2
      ;;
    -b_n)
      BUCKET_NAME="$2"
      shift 2
      ;;
    -c_n)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    -region)
      REGION="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

# Validate mandatory arguments
if [ -z "$PROJECT_ID" ]; then
    usage
fi

# Delete project function
delete_project() {
    echo "Deleting the GCP project..."
    if gcloud projects describe "$PROJECT_ID" > /dev/null 2>&1; then
        gcloud projects delete "$PROJECT_ID" --quiet
        echo "Project $PROJECT_ID deleted successfully."
    else
        echo "Project $PROJECT_ID does not exist or is already deleted."
    fi
}

# Delete bucket function
delete_bucket() {
    echo "Deleting the Google Cloud Storage bucket..."
    if gsutil ls -b "gs://$BUCKET_NAME" > /dev/null 2>&1; then
        gsutil -m rm -r "gs://$BUCKET_NAME"
        echo "Bucket $BUCKET_NAME deleted successfully."
    else
        echo "Bucket $BUCKET_NAME does not exist or is already deleted."
    fi
}

# Delete cluster function
delete_cluster() {
    echo "Deleting the Dataproc cluster in region $REGION..."
    if gcloud dataproc clusters describe "$CLUSTER_NAME" --region "$REGION" > /dev/null 2>&1; then
        gcloud dataproc clusters delete "$CLUSTER_NAME" --region="$REGION" --quiet
        echo "Cluster $CLUSTER_NAME deleted successfully."
    else
        echo "Cluster $CLUSTER_NAME does not exist or is already deleted."
    fi
}

# Determine what to delete
if [ "$DELETE_PROJECT" = false ] && [ "$DELETE_BUCKET" = false ] && [ "$DELETE_CLUSTER" = false ]; then
    # Default: Delete everything
    DELETE_PROJECT=true
    DELETE_BUCKET=true
    DELETE_CLUSTER=true
fi

# Execute the appropriate functions
if [ "$DELETE_CLUSTER" = true ]; then
    delete_cluster
fi

if [ "$DELETE_BUCKET" = true ]; then
    delete_bucket
fi

if [ "$DELETE_PROJECT" = true ]; then
    delete_project
fi

echo "Requested deletions completed successfully."
