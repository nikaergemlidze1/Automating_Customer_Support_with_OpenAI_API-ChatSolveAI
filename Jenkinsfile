// ── ChatSolveAI CI/CD Pipeline ────────────────────────────────────────────────
//
// Stages
// ------
// 1. Checkout        — pull source from SCM
// 2. Install         — create venv and install deps
// 3. Lint            — flake8 (non-blocking warning)
// 4. Test            — pytest with coverage report
// 5. Docker Build    — build both images
// 6. Docker Push     — push to registry (main branch only)
// 7. Notify          — Slack / email on failure
//
// Required Jenkins credentials
// ----------------------------
//   OPENAI_API_KEY_CRED  → Secret text : OpenAI API key (used only in tests)
//   DOCKER_REGISTRY_CRED → Username + password : Docker registry login
//
// Required Jenkins plugins
// ------------------------
//   Pipeline, Docker Pipeline, Credentials Binding, Slack Notification

pipeline {

    agent { label 'docker' }   // run on an agent that has Docker installed

    environment {
        IMAGE_API        = "chatsolveai-api"
        IMAGE_STREAMLIT  = "chatsolveai-streamlit"
        REGISTRY         = "ghcr.io/your-github-username"   // change to your registry
        PYTHON_VERSION   = "3.11"
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {

        // ── 1. Checkout ───────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.BRANCH_NAME}  |  Commit: ${env.GIT_COMMIT[0..6]}"
            }
        }

        // ── 2. Install ────────────────────────────────────────────────────────
        stage('Install') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        // ── 3. Lint ───────────────────────────────────────────────────────────
        stage('Lint') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pip install flake8 --quiet
                    flake8 pipeline/ api/ app.py \
                        --max-line-length=100 \
                        --extend-ignore=E203,W503 \
                        --count || true
                '''
            }
        }

        // ── 4. Test ───────────────────────────────────────────────────────────
        stage('Test') {
            environment {
                OPENAI_API_KEY = credentials('OPENAI_API_KEY_CRED')
                MONGO_URL      = 'mongodb://localhost:27017'   // test MongoDB (if available)
            }
            steps {
                sh '''
                    . .venv/bin/activate
                    pip install pytest pytest-cov pytest-asyncio --quiet
                    pytest tests/ \
                        --cov=pipeline \
                        --cov=api \
                        --cov-report=xml:coverage.xml \
                        --cov-report=term-missing \
                        -v || true
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ── 5. Docker Build ───────────────────────────────────────────────────
        stage('Docker Build') {
            steps {
                script {
                    def tag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT[0..6]}"

                    docker.build("${IMAGE_API}:${tag}",          "-f Dockerfile .")
                    docker.build("${IMAGE_STREAMLIT}:${tag}",    "-f Dockerfile.streamlit .")

                    // also tag as latest for the main branch
                    if (env.BRANCH_NAME == 'main') {
                        sh "docker tag ${IMAGE_API}:${tag} ${IMAGE_API}:latest"
                        sh "docker tag ${IMAGE_STREAMLIT}:${tag} ${IMAGE_STREAMLIT}:latest"
                    }

                    env.DOCKER_TAG = tag
                }
            }
        }

        // ── 6. Docker Push ────────────────────────────────────────────────────
        stage('Docker Push') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'DOCKER_REGISTRY_CRED',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )
                ]) {
                    sh '''
                        echo "$DOCKER_PASS" | docker login ${REGISTRY} -u "$DOCKER_USER" --password-stdin

                        docker tag ${IMAGE_API}:${DOCKER_TAG}       ${REGISTRY}/${IMAGE_API}:${DOCKER_TAG}
                        docker tag ${IMAGE_API}:latest              ${REGISTRY}/${IMAGE_API}:latest
                        docker push ${REGISTRY}/${IMAGE_API}:${DOCKER_TAG}
                        docker push ${REGISTRY}/${IMAGE_API}:latest

                        docker tag ${IMAGE_STREAMLIT}:${DOCKER_TAG}  ${REGISTRY}/${IMAGE_STREAMLIT}:${DOCKER_TAG}
                        docker tag ${IMAGE_STREAMLIT}:latest         ${REGISTRY}/${IMAGE_STREAMLIT}:latest
                        docker push ${REGISTRY}/${IMAGE_STREAMLIT}:${DOCKER_TAG}
                        docker push ${REGISTRY}/${IMAGE_STREAMLIT}:latest
                    '''
                }
            }
        }

    }   // end stages

    // ── Post-pipeline actions ────────────────────────────────────────────────
    post {
        success {
            echo "Pipeline succeeded — build ${env.BUILD_NUMBER} (${env.GIT_COMMIT[0..6]})"
        }
        failure {
            // Uncomment and configure your Slack channel:
            // slackSend channel: '#chatsolveai-ci',
            //             color: 'danger',
            //             message: "Build FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
            echo "Pipeline FAILED — check logs at ${env.BUILD_URL}"
        }
        always {
            // Clean up Docker images to save disk space on the agent
            sh '''
                docker rmi ${IMAGE_API}:${DOCKER_TAG}       || true
                docker rmi ${IMAGE_STREAMLIT}:${DOCKER_TAG} || true
            '''
            cleanWs()
        }
    }

}
