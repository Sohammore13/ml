import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages after page 1)
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)
            self.drawString(36, 760, "Machine Learning Lab — Complete Theory & Code Manual (CO1 - CO6)")
            self.drawRightString(576, 760, "Experiments 1 to 12")

        # Footer on all pages
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_str)
        self.drawString(36, 30, "Prepared for Google Colab / Practical Lab Exam Revision")
        self.restoreState()

# Theory content dictionary for all 12 experiments
EXPERIMENTS_DATA = [
    {
        "id": 1,
        "co": "CO1",
        "title": "Complete ML Pipeline & Data Preprocessing",
        "objective": "To understand and implement the complete machine learning pipeline from raw data handling, imputation, categorical encoding, feature scaling to train-test splitting.",
        "theory": (
            "<b>1. Machine Learning Pipeline Architecture:</b> An ML pipeline chains raw data extraction, cleaning, feature transformation, model training, and evaluation into a seamless, reproducible workflow.<br/>"
            "<b>2. Missing Value Imputation:</b> Real-world data often lacks complete observations. Mean/median imputation replaces missing entries with column statistics to prevent model bias.<br/>"
            "<b>3. Feature Scaling (Standardization):</b> Algorithms sensitive to distance (SVM, KNN, Logistic Regression, Neural Networks) require standardized features where mean &mu; = 0 and variance &sigma;<sup>2</sup> = 1:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<b>Formula:</b> <i>z = (x - &mu;) / &sigma;</i><br/>"
            "<b>4. Train-Test Split:</b> Partitioning data (typically 80% train, 20% test) prevents data leakage and ensures unbiased evaluation of generalization capability on unseen samples."
        ),
        "code_file": "1.py",
        "output_notes": "Outputs feature distributions before and after scaling using side-by-side boxplots, verifying zero-mean unit-variance normalization."
    },
    {
        "id": 2,
        "co": "CO1",
        "title": "Model Evaluation using a Classification Task",
        "objective": "To evaluate classification models using performance metrics including Confusion Matrix, Accuracy, Precision, Recall, and F1-score.",
        "theory": (
            "<b>1. The Confusion Matrix:</b> A 2D matrix summarizing model predictions vs actual ground truth:<br/>"
            "&nbsp;&nbsp;&bull; <b>TP (True Positive):</b> Correct positive predictions | <b>TN (True Negative):</b> Correct negative predictions<br/>"
            "&nbsp;&nbsp;&bull; <b>FP (False Positive / Type I error):</b> Negative predicted as positive | <b>FN (False Negative / Type II error):</b> Positive predicted as negative.<br/>"
            "<b>2. Mathematical Evaluation Formulas:</b><br/>"
            "&nbsp;&nbsp;&bull; <b>Accuracy:</b> <i>(TP + TN) / (TP + TN + FP + FN)</i> &mdash; Overall fraction of correct predictions.<br/>"
            "&nbsp;&nbsp;&bull; <b>Precision:</b> <i>TP / (TP + FP)</i> &mdash; Fraction of predicted positives that are truly positive.<br/>"
            "&nbsp;&nbsp;&bull; <b>Recall (Sensitivity):</b> <i>TP / (TP + FN)</i> &mdash; Fraction of actual positives identified.<br/>"
            "&nbsp;&nbsp;&bull; <b>F1-Score:</b> <i>2 &times; (Precision &times; Recall) / (Precision + Recall)</i> &mdash; Harmonic mean balancing Precision and Recall."
        ),
        "code_file": "2.py",
        "output_notes": "Displays detailed Classification Report table and renders a color-coded Confusion Matrix heatmap."
    },
    {
        "id": 3,
        "co": "CO1",
        "title": "Model Evaluation using K-Fold Cross-Validation & Bias-Variance Curves",
        "objective": "To implement K-Fold Cross Validation for variance reduction and diagnose bias-variance tradeoff using Learning and Validation curves.",
        "theory": (
            "<b>1. K-Fold Cross-Validation:</b> The dataset is split into <i>k</i> equal folds. In each iteration, <i>k-1</i> folds train the model while the remaining fold validates it. The final score is the mean of all <i>k</i> runs, minimizing variance caused by single random splits.<br/>"
            "<b>2. Bias-Variance Tradeoff:</b><br/>"
            "&nbsp;&nbsp;&bull; <b>High Bias (Underfitting):</b> Model is too simple; both training and validation errors remain high.<br/>"
            "&nbsp;&nbsp;&bull; <b>High Variance (Overfitting):</b> Model memorizes training noise; training error is low but validation error is high.<br/>"
            "<b>3. Diagnostic Curves:</b><br/>"
            "&nbsp;&nbsp;&bull; <b>Learning Curve:</b> Plots training & validation score vs training sample size.<br/>"
            "&nbsp;&nbsp;&bull; <b>Validation Curve:</b> Plots training & validation score vs a hyperparameter (e.g., <i>max_depth</i>) to identify the optimal complexity sweet-spot."
        ),
        "code_file": "3.py",
        "output_notes": "Plots side-by-side Learning Curve and Validation Curve showing convergence and hyperparameter tuning sweet spots."
    },
    {
        "id": 4,
        "co": "CO3",
        "title": "Gradient Descent Algorithm for Cost Function Optimization",
        "objective": "To implement first-order Gradient Descent optimization from scratch to minimize Mean Squared Error (MSE) cost function.",
        "theory": (
            "<b>1. Concept:</b> Gradient Descent iteratively updates parameters in the direction of the steepest descent (negative gradient) of the cost function <i>J(m, c)</i>.<br/>"
            "<b>2. Cost Function (Mean Squared Error):</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>J(m, c) = (1/n) &Sigma; [y<sub>i</sub> - (m&middot;x<sub>i</sub> + c)]<sup>2</sup></i><br/>"
            "<b>3. Gradient Partial Derivatives:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>&part;J/&part;m = (-2/n) &Sigma; x<sub>i</sub> &middot; [y<sub>i</sub> - &ycirc;<sub>i</sub>]</i><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>&part;J/&part;c = (-2/n) &Sigma; [y<sub>i</sub> - &ycirc;<sub>i</sub>]</i><br/>"
            "<b>4. Parameter Update Rule:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>m &larr; m - &alpha; &middot; (&part;J/&part;m)</i> &nbsp;&nbsp;|&nbsp;&nbsp; <i>c &larr; c - &alpha; &middot; (&part;J/&part;c)</i><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;where <i>&alpha;</i> is the learning rate (step size)."
        ),
        "code_file": "4.py",
        "output_notes": "Plots cost vs epochs demonstrating monotonic convergence, alongside a scatter plot with the fitted regression line."
    },
    {
        "id": 5,
        "co": "CO3",
        "title": "Linear Regression with L1 (Lasso) and L2 (Ridge) Regularization",
        "objective": "To implement Linear Regression and apply L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting and handle multicollinearity.",
        "theory": (
            "<b>1. Motivation for Regularization:</b> Ordinary Least Squares (OLS) minimizes training residual error but can overfit when features are collinear or numerous.<br/>"
            "<b>2. Ridge Regression (L2 Regularization):</b> Adds a squared penalty term:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>Cost<sub>Ridge</sub> = &Sigma; (y<sub>i</sub> - &ycirc;<sub>i</sub>)<sup>2</sup> + &lambda; &Sigma; w<sub>j</sub><sup>2</sup></i><br/>"
            "&nbsp;&nbsp;&bull; Shrinks coefficients smoothly towards zero but never exact zero. Excellent for handling multicollinearity.<br/>"
            "<b>3. Lasso Regression (L1 Regularization):</b> Adds an absolute penalty term:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>Cost<sub>Lasso</sub> = &Sigma; (y<sub>i</sub> - &ycirc;<sub>i</sub>)<sup>2</sup> + &lambda; &Sigma; |w<sub>j</sub>|</i><br/>"
            "&nbsp;&nbsp;&bull; Forces insignificant feature coefficients to become exactly 0, performing automatic <b>feature selection</b>."
        ),
        "code_file": "5.py",
        "output_notes": "Compares Mean Squared Error across Standard Linear Regression, Lasso (L1), and Ridge (L2) using a clear bar chart."
    },
    {
        "id": 6,
        "co": "CO3",
        "title": "Support Vector Machines (SVM) for Classification",
        "objective": "To implement and analyze Support Vector Machine classifiers with Linear and Radial Basis Function (RBF) non-linear kernels.",
        "theory": (
            "<b>1. Fundamental Concept:</b> SVM identifies the optimal hyperplane <i>w<sup>T</sup>x + b = 0</i> that maximizes the geometric margin <i>2 / ||w||</i> between the nearest points of distinct classes (<b>Support Vectors</b>).<br/>"
            "<b>2. Soft Margin & Regularization (C parameter):</b> Balances margin maximization against classification error penalty.<br/>"
            "<b>3. The Kernel Trick:</b> Projects non-linearly separable input data into a higher-dimensional space where linear separation becomes feasible without explicitly computing high-dimensional coordinates:<br/>"
            "&nbsp;&nbsp;&bull; <b>Linear Kernel:</b> <i>K(x, x') = x<sup>T</sup> x'</i> (best for linearly separable data)<br/>"
            "&nbsp;&nbsp;&bull; <b>RBF (Gaussian) Kernel:</b> <i>K(x, x') = exp(-&gamma; ||x - x'||<sup>2</sup>)</i> (effective for complex non-linear boundaries)."
        ),
        "code_file": "6.py",
        "output_notes": "Outputs test classification metrics and renders an SVM confusion matrix display."
    },
    {
        "id": 7,
        "co": "CO3",
        "title": "Logistic Regression for Binary Classification",
        "objective": "To implement Logistic Regression for binary classification and evaluate probabilistic predictions using ROC-AUC curves.",
        "theory": (
            "<b>1. Sigmoid Hypothesis:</b> Maps any real-valued linear input <i>z = w<sup>T</sup>x + b</i> into a valid probability range <i>[0, 1]</i>:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>P(y = 1 | x) = &sigma;(z) = 1 / (1 + e<sup>-z</sup>)</i><br/>"
            "<b>2. Decision Rule:</b> Predict class 1 if <i>&sigma;(z) &ge; 0.5</i>; otherwise predict class 0.<br/>"
            "<b>3. Binary Cross-Entropy Loss:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>Loss = - (1/n) &Sigma; [y<sub>i</sub> log(&ycirc;<sub>i</sub>) + (1 - y<sub>i</sub>) log(1 - &ycirc;<sub>i</sub>)]</i><br/>"
            "<b>4. ROC & AUC Analysis:</b> Receiver Operating Characteristic plots True Positive Rate vs False Positive Rate across all thresholds. AUC &approx; 1.0 indicates perfect discrimination."
        ),
        "code_file": "7.py",
        "output_notes": "Displays side-by-side Confusion Matrix and ROC Curve demonstrating class discriminative power."
    },
    {
        "id": 8,
        "co": "CO4",
        "title": "Logic Gates Implementation using McCulloch-Pitts (MP) Neurons",
        "objective": "To implement AND, OR, NOT, and NAND logic gates using the foundational McCulloch-Pitts mathematical neuron model.",
        "theory": (
            "<b>1. Historical Context:</b> Proposed in 1943 by Warren McCulloch and Walter Pitts, this was the earliest mathematical abstraction of biological neurons.<br/>"
            "<b>2. Mathematical Formulation:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>Output = 1 &nbsp;if&nbsp; &Sigma; (x<sub>i</sub> &middot; w<sub>i</sub>) &ge; &theta; &nbsp;else&nbsp; 0</i><br/>"
            "&nbsp;&nbsp;&bull; Inputs <i>x<sub>i</sub> &isin; {0, 1}</i>. Weights are excitatory (+1) or inhibitory (-1).<br/>"
            "<b>3. Logic Gate Configurations:</b><br/>"
            "&nbsp;&nbsp;&bull; <b>AND Gate:</b> <i>w = [1, 1]</i>, threshold <i>&theta; = 2</i> (fires only when both inputs are active).<br/>"
            "&nbsp;&nbsp;&bull; <b>OR Gate:</b> <i>w = [1, 1]</i>, threshold <i>&theta; = 1</i> (fires when at least one input is active).<br/>"
            "&nbsp;&nbsp;&bull; <b>NOT Gate:</b> <i>w = [-1]</i>, threshold <i>&theta; = 0</i> (inhibitory gate inverting single input).<br/>"
            "<b>4. Fundamental Limitation:</b> MP neurons only solve linearly separable functions; cannot solve XOR without hidden layers."
        ),
        "code_file": "8.py",
        "output_notes": "Visualizes 2D truth-table coordinate points and draws explicit linear decision boundaries for AND and OR gates."
    },
    {
        "id": 9,
        "co": "CO4",
        "title": "Multi-Layer Perceptron (MLP) with Backpropagation Weight Changes",
        "objective": "To implement an MLP network from scratch and record explicit weight changes (&Delta;W) computed via gradient backpropagation.",
        "theory": (
            "<b>1. Architecture:</b> Input Layer (2 nodes) &rarr; Hidden Layer (2 nodes) &rarr; Output Layer (1 node). Overcomes linear inseparability.<br/>"
            "<b>2. Forward Pass:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>h = &sigma;(X &middot; W<sub>1</sub>)</i> &nbsp;&nbsp;|&nbsp;&nbsp; <i>&ycirc; = &sigma;(h &middot; W<sub>2</sub>)</i> &nbsp;&nbsp;(where <i>&sigma;(z) = 1 / (1 + e<sup>-z</sup>)</i>)<br/>"
            "<b>3. Backward Pass (Chain Rule Gradients):</b><br/>"
            "&nbsp;&nbsp;&bull; Output Gradient: <i>&delta;<sub>out</sub> = (y - &ycirc;) &odot; &sigma;'(&ycirc;) = (y - &ycirc;) &odot; &ycirc;(1 - &ycirc;)</i><br/>"
            "&nbsp;&nbsp;&bull; Hidden Gradient: <i>&delta;<sub>hidden</sub> = (&delta;<sub>out</sub> &middot; W<sub>2</sub><sup>T</sup>) &odot; h(1 - h)</i><br/>"
            "<b>4. Weight Updates:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>&Delta;W<sub>2</sub> = h<sup>T</sup> &middot; &delta;<sub>out</sub> &middot; lr</i> &nbsp;&nbsp;|&nbsp;&nbsp; <i>&Delta;W<sub>1</sub> = X<sup>T</sup> &middot; &delta;<sub>hidden</sub> &middot; lr</i><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>W<sub>2</sub> &larr; W<sub>2</sub> + &Delta;W<sub>2</sub></i> &nbsp;&nbsp;|&nbsp;&nbsp; <i>W<sub>1</sub> &larr; W<sub>1</sub> + &Delta;W<sub>1</sub></i>"
        ),
        "code_file": "9.py",
        "output_notes": "Prints numerical weight update tables (&Delta;W) across training epochs and plots the error loss reduction curve."
    },
    {
        "id": 10,
        "co": "CO4",
        "title": "Supervised Hebbian Learning Rule for a Single Neuron",
        "objective": "To implement the supervised Hebbian learning rule for binary classification and track step-by-step weight changes.",
        "theory": (
            "<b>1. Hebb's Postulate (1949):</b> 'Neurons that fire together, wire together.' If two connected neurons are simultaneously active, synaptic strength increases.<br/>"
            "<b>2. Supervised Learning Formulation:</b> In supervised classification, the weight adjustment is proportional to the product of the input and the target output:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>&Delta;w<sub>i</sub> = x<sub>i</sub> &middot; y</i> &nbsp;&nbsp;|&nbsp;&nbsp; <i>&Delta;b = y</i><br/>"
            "<b>3. Weight and Bias Update Rules:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>w<sub>i</sub><sup>(new)</sup> = w<sub>i</sub><sup>(old)</sup> + &Delta;w<sub>i</sub></i> &nbsp;&nbsp;|&nbsp;&nbsp; <i>b<sup>(new)</sup> = b<sup>(old)</sup> + &Delta;b</i><br/>"
            "<b>4. Importance of Bipolar Representation:</b> Using bipolar values <i>x<sub>i</sub>, y &isin; {-1, +1}</i> ensures weights decrease when patterns mismatch, guaranteeing proper decision boundary orientation."
        ),
        "code_file": "10.py",
        "output_notes": "Prints step-by-step table tracking input, target, weight change (&Delta;w), and renders the final 2D decision boundary line."
    },
    {
        "id": 11,
        "co": "CO5",
        "title": "Dimensionality Reduction using PCA and t-SNE",
        "objective": "To implement and contrast linear (PCA) and non-linear (t-SNE) dimensionality reduction techniques on high-dimensional data.",
        "theory": (
            "<b>1. The Curse of Dimensionality:</b> As feature dimensions increase, data sparsity escalates, distance metrics lose discriminative power, and computation costs surge.<br/>"
            "<b>2. Principal Component Analysis (PCA - Linear):</b><br/>"
            "&nbsp;&nbsp;&bull; Finds orthogonal axes (Principal Components) that maximize variance.<br/>"
            "&nbsp;&nbsp;&bull; Computed via Eigen-decomposition of the feature covariance matrix <i>&Sigma; = (1/n) X<sup>T</sup> X</i>.<br/>"
            "&nbsp;&nbsp;&bull; Preserves <b>global data structure</b> and variance; deterministic and fast.<br/>"
            "<b>3. t-Distributed Stochastic Neighbor Embedding (t-SNE - Non-Linear):</b><br/>"
            "&nbsp;&nbsp;&bull; Converts pairwise Euclidean distances into joint conditional probabilities using Gaussian (high-dim) and Student-t (low-dim) distributions.<br/>"
            "&nbsp;&nbsp;&bull; Minimizes Kullback-Leibler (KL) divergence via gradient descent.<br/>"
            "&nbsp;&nbsp;&bull; Preserves <b>local cluster manifolds</b>; ideal for visual exploration of complex clusters."
        ),
        "code_file": "11.py",
        "output_notes": "Displays side-by-side 2D scatter plots comparing global variance preservation in PCA vs cluster separation in t-SNE."
    },
    {
        "id": 12,
        "co": "CO6",
        "title": "Foundational Concepts of MLOps & Automated End-to-End Pipeline",
        "objective": "To introduce foundational MLOps principles: environment setup, Git versioning, and building an automated end-to-end ML pipeline.",
        "theory": (
            "<b>1. What is MLOps?</b> MLOps (Machine Learning Operations) unites ML development (Dev) with operations (Ops) to automate and monitor the entire ML model lifecycle.<br/>"
            "<b>2. Core Foundational Pillars:</b><br/>"
            "&nbsp;&nbsp;&bull; <b>Environment Reproducibility:</b> Pinning dependencies using <i>requirements.txt</i> and virtual environments (<i>python -m venv</i>).<br/>"
            "&nbsp;&nbsp;&bull; <b>Version Control with Git:</b> Tracking code versions, commit histories, branching strategies, and data versioning.<br/>"
            "&nbsp;&nbsp;&bull; <b>Automated Pipeline (CI/CD):</b> Using <i>sklearn.pipeline.Pipeline</i> to bundle preprocessing (scaling) and model training into an atomic, leak-free object.<br/>"
            "&nbsp;&nbsp;&bull; <b>Model Registry & Persistence:</b> Serializing trained pipeline artifacts with <i>joblib</i> or <i>MLflow</i> for production deployment.<br/>"
            "&nbsp;&nbsp;&bull; <b>Monitoring:</b> Logging pipeline latency, execution states, and tracking distribution drift in production."
        ),
        "code_file": "12.py",
        "output_notes": "Demonstrates structured logging, automated pipeline execution, artifact export (joblib), and inference verification."
    }
]

def build_comprehensive_pdf():
    pdf_filename = "Machine_Learning_Lab_Manual_With_Theory.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=12
    )

    pattern_box_style = ParagraphStyle(
        'PatternBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    exp_heading_style = ParagraphStyle(
        'ExpHeading',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.white
    )

    co_tag_style = ParagraphStyle(
        'COTag',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#38bdf8'),
        alignment=2
    )

    section_label_style = ParagraphStyle(
        'SectionLabel',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'ExpBody',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        fontName='Courier',
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Machine Learning Lab Manual & Exam Revision Guide", title_style))
    story.append(Paragraph("Comprehensive Theoretical Foundations, Mathematical Formulations & Exam-Ready Python Implementations", subtitle_style))

    # Master 7-Step Pattern Box
    pattern_html = (
        "<b>THE UNIVERSAL 7-STEP PATTERN TO MEMORIZE FOR EVERY LAB EXPERIMENT:</b><br/>"
        "<b>1. Import libraries:</b> sklearn datasets/models/metrics, numpy, matplotlib.pyplot<br/>"
        "<b>2. Data Collection:</b> Built-in datasets (load_iris, load_breast_cancer, load_diabetes, load_digits, make_regression)<br/>"
        "<b>3. Data Preprocessing:</b> StandardScaler().fit_transform(X)<br/>"
        "<b>4. Split Dataset:</b> train_test_split(X, y, test_size=0.2, random_state=42)<br/>"
        "<b>5. Model Training:</b> model = EstimatorClass(); model.fit(X_train, y_train)<br/>"
        "<b>6. Prediction:</b> y_pred = model.predict(X_test)<br/>"
        "<b>7. Evaluation & Visualization:</b> Metrics (accuracy_score, confusion_matrix, classification_report) & plt.show()"
    )
    pattern_table = Table([[Paragraph(pattern_html, pattern_box_style)]], colWidths=[540])
    pattern_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3b82f6')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(pattern_table)
    story.append(Spacer(1, 10))

    # Iterate through all 12 experiments
    for exp in EXPERIMENTS_DATA:
        exp_id = exp["id"]
        exp_co = exp["co"]
        exp_title = exp["title"]
        exp_obj = exp["objective"]
        exp_theory = exp["theory"]
        exp_file = exp["code_file"]
        exp_notes = exp["output_notes"]

        # Read Code from individual file
        with open(exp_file, 'r') as f:
            code_text = f.read().strip()

        # Clean title banner table
        header_row = [
            Paragraph(f"<b>Experiment {exp_id}: {exp_title}</b>", exp_heading_style),
            Paragraph(f"<b>[{exp_co}]</b>", co_tag_style)
        ]
        header_table = Table([header_row], colWidths=[460, 80])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Objective block
        obj_p = Paragraph(f"<b>Aim / Objective:</b> {exp_obj}", body_style)

        # Theory block
        theory_title = Paragraph("<b>Theoretical Background & Mathematical Formulation:</b>", section_label_style)
        theory_p = Paragraph(exp_theory, body_style)

        theory_box = Table([[Paragraph(f"<b>Objective:</b> {exp_obj}<br/><br/><b>Theory & Key Concepts:</b><br/>{exp_theory}", body_style)]], colWidths=[540])
        theory_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        # Code block
        code_label = Paragraph("<b>Exam-Ready Python Implementation (7-Step Pattern):</b>", section_label_style)
        code_p = Preformatted(code_text, code_style)
        code_box = Table([[code_p]], colWidths=[540])
        code_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        # Expected output block
        notes_p = Paragraph(f"<b>Expected Output & Graph:</b> {exp_notes}", body_style)

        # Assemble experiment elements
        exp_elements = [
            header_table,
            Spacer(1, 4),
            theory_box,
            Spacer(1, 4),
            code_label,
            code_box,
            Spacer(1, 3),
            notes_p,
            Spacer(1, 12)
        ]

        # Use PageBreak after each major experiment for clean printing if desired, or keep together
        story.append(KeepTogether(exp_elements[:3])) # Header + theory together
        story.append(Spacer(1, 4))
        story.append(code_label)
        story.append(code_box)
        story.append(Spacer(1, 3))
        story.append(notes_p)
        story.append(Spacer(1, 14))

    doc.build(story, canvasmaker=NumberedCanvas)
    print("Successfully built: Machine_Learning_Lab_Manual_With_Theory.pdf")

if __name__ == '__main__':
    build_comprehensive_pdf()
