#!/bin/bash

# Copyright (c) 2025. All rights reserved.
# DS 5460: Big Data Scaling
# Spring 2025
# Author: Yehyun Suh; yehyun.suh@vanderbilt.edu
# Description: This script automates the creation of Google Cloud resources, including projects, buckets, and clusters.
# License: This script is provided "as-is" without warranty of any kind. Use at your own risk.

# Example Usage
# This script allows you to create Google Cloud resources using the following commands:
#
# 1. Create all resources (project, bucket, and cluster):
#    bash create_gc_bash.sh -p_id PROJECT_ID -b_n BUCKET_NAME -c_n CLUSTER_NAME -region REGION -bill BILLING_ACCOUNT 
#    Example: bash create_gc_bash.sh -p_id my-project -b_n my-bucket -c_n my-cluster -region us-central1 -bill my-billing-account
#
# 2. Create only a project:
#    bash create_gc_bash.sh -p -p_id PROJECT_ID -bill BILLING_ACCOUNT
#    Example: bash create_gc_bash.sh -p -p_id my-project -bill my-billing-account
#
# 3. Create only a bucket:
#    bash create_gc_bash.sh -b -p_id PROJECT_ID -b_n BUCKET_NAME [-region REGION]
#    Example: bash create_gc_bash.sh -b -p_id my-project -b_n my-bucket -region us-central1
#
# 4. Create only a cluster:
#    bash create_gc_bash.sh -c -p_id PROJECT_ID -b_n BUCKET_NAME -c_n CLUSTER_NAME [-region REGION]
#    Example: bash create_gc_bash.sh -c -p_id my-project -b_n my-bucket -c_n my-cluster -region us-central1
#
# 5. Create a bucket and cluster:
#    bash create_gc_bash.sh -b -c -p_id PROJECT_ID -b_n BUCKET_NAME -c_n CLUSTER_NAME [-region REGION]
#    Example: bash create_gc_bash.sh -bc -p_id my-project -b_n my-bucket -c_n my-cluster -region us-central1
#
# Notes:
# - PROJECT_ID is mandatory for all operations.
# - BUCKET_NAME is required for operations involving buckets or clusters.
# - CLUSTER_NAME is required for operations involving clusters.
# - BILLING_ACCOUNT is required for creating a project.
# - REGION is optional and defaults to us-central1.


# Usage information
usage() {
    echo "Usage: $0 [OPTIONS] -p_id PROJECT_ID [-b_n BUCKET_NAME] [-c_n CLUSTER_NAME] [-region REGION] [-bill BILLING_ACCOUNT]"
    echo "Options:"
    echo "  -p             Create project only"
    echo "  -b             Create bucket only"
    echo "  -c             Create cluster only"
    echo "  -bc            Create bucket and cluster"
    echo "  -pbc           Create project, bucket, and cluster"
    echo "No options:      Create project, bucket, and cluster"
    exit 1
}

# Default flags
CREATE_PROJECT=false
CREATE_BUCKET=false
CREATE_CLUSTER=false
REGION="us-central1"  # Default region

# Parse options
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -p)
      CREATE_PROJECT=true
      shift
      ;;
    -b)
      CREATE_BUCKET=true
      shift
      ;;
    -c)
      CREATE_CLUSTER=true
      shift
      ;;
    -bill)
      BILLING_ACCOUNT="$2"
      shift 2
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

# Create project function
create_project() {
    if ! gcloud projects describe "$PROJECT_ID" > /dev/null 2>&1; then
        echo "Creating Google Cloud project..."
        gcloud projects create "$PROJECT_ID" --name="$PROJECT_ID"
        gcloud config set project "$PROJECT_ID"

        if [ -n "$BILLING_ACCOUNT" ]; then
            echo "Linking billing account to the project..."
            gcloud beta billing projects link "$PROJECT_ID" --billing-account "$BILLING_ACCOUNT"
        fi

        echo "Enabling APIs..."
        gcloud services enable cloudresourcemanager.googleapis.com
        gcloud services enable dataproc.googleapis.com
        gcloud services enable compute.googleapis.com
        gcloud services enable storage.googleapis.com

        echo "Granting Storage Admin role to Compute Engine service account..."
        COMPUTE_ENGINE_SERVICE_ACCOUNT=$(gcloud iam service-accounts list \
            --filter="displayName:'Compute Engine default service account'" \
            --format 'value(email)')
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member "serviceAccount:$COMPUTE_ENGINE_SERVICE_ACCOUNT" \
            --role roles/storage.admin

        echo "Project setup completed successfully."
    else
        echo "Project $PROJECT_ID already exists."
    fi
}

# Create bucket function
create_bucket() {
    echo "Creating Google Cloud Storage bucket..."
    if ! gsutil ls -b "gs://$BUCKET_NAME" > /dev/null 2>&1; then
        gcloud storage buckets create "gs://$BUCKET_NAME" --location="$REGION" --uniform-bucket-level-access
        echo "Bucket $BUCKET_NAME created successfully."
    else
        echo "Bucket $BUCKET_NAME already exists."
    fi
}

# Create cluster function
create_cluster() {
    echo "Creating Dataproc cluster..."
    if ! gcloud dataproc clusters describe "$CLUSTER_NAME" --region="$REGION" > /dev/null 2>&1; then
        gcloud dataproc clusters create "$CLUSTER_NAME" \
            --region="$REGION" \
            --single-node \
            --image-version=1.5 \
            --optional-components=ANACONDA,JUPYTER \
            --enable-component-gateway \
            --bucket="$BUCKET_NAME"
        echo "Cluster $CLUSTER_NAME created successfully."
    else
        echo "Cluster $CLUSTER_NAME already exists."
    fi
}

# Determine what to create
if [ "$CREATE_PROJECT" = false ] && [ "$CREATE_BUCKET" = false ] && [ "$CREATE_CLUSTER" = false ]; then
    # Default: Create everything
    CREATE_PROJECT=true
    CREATE_BUCKET=true
    CREATE_CLUSTER=true
fi

if [ "$CREATE_BUCKET" = true ] && [ "$CREATE_CLUSTER" = true ] && [ "$CREATE_PROJECT" = false ]; then
    CREATE_PROJECT=false
    CREATE_BUCKET=true
    CREATE_CLUSTER=true
fi

# Execute the appropriate functions
if [ "$CREATE_PROJECT" = true ]; then
    create_project
fi

if [ "$CREATE_BUCKET" = true ]; then
    create_bucket
fi

if [ "$CREATE_CLUSTER" = true ]; then
    create_cluster
fi

echo "Requested operations completed successfully."
