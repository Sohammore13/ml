from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfgen import canvas
import sys

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
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        # Header line & text
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(40, 755, 572, 755)
        self.drawString(40, 760, "Machine Learning Lab Experiments — Exam Revision Manual")
        
        # Footer line & text
        self.line(40, 45, 572, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 32, page_str)
        self.drawString(40, 32, "Prepared for Google Colab / Practical Lab Exam")
        self.restoreState()

def build_pdf():
    pdf_filename = "Machine_Learning_Lab_Manual.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=55,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1, # Center
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        alignment=1,
        spaceAfter=15
    )

    pattern_title_style = ParagraphStyle(
        'PatternTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=4
    )

    pattern_text_style = ParagraphStyle(
        'PatternText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b')
    )

    exp_heading_style = ParagraphStyle(
        'ExpHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.white,
        spaceAfter=0
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Machine Learning Lab Manual", title_style))
    story.append(Paragraph("Complete Implementation of Experiments 1 to 10 with Library Datasets & Graphs", subtitle_style))

    # Pattern Box Table
    pattern_content = [
        Paragraph("<b>The Universal 7-Step Pattern to Memorize:</b>", pattern_title_style),
        Paragraph("<b>1. Import libraries:</b> sklearn, numpy, matplotlib.pyplot<br/>"
                  "<b>2. Data Collection:</b> Built-in datasets (load_iris, load_breast_cancer, load_diabetes, make_regression)<br/>"
                  "<b>3. Data Preprocessing:</b> StandardScaler().fit_transform(X)<br/>"
                  "<b>4. Split the dataset:</b> train_test_split(X, y, test_size=0.2, random_state=42)<br/>"
                  "<b>5. Model Training:</b> model = ModelClass(); model.fit(X_train, y_train)<br/>"
                  "<b>6. Prediction:</b> y_pred = model.predict(X_test)<br/>"
                  "<b>7. Evaluation & Visualization:</b> accuracy_score, confusion_matrix & plt.show()", pattern_text_style)
    ]
    pattern_table = Table([[pattern_content]], colWidths=[532])
    pattern_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f9ff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#38bdf8')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(pattern_table)
    story.append(Spacer(1, 15))

    # Read all experiments from all_codes.py
    with open('all_codes.py', 'r') as f:
        content = f.read()

    experiments = content.split('# ==============================================================================')

    for exp in experiments[1:]:
        exp = exp.strip()
        if not exp:
            continue
        lines = exp.split('\n')
        title = lines[0].replace('#', '').strip()
        
        # Remove print statements for clean reading in PDF
        code_lines = []
        for line in lines[1:]:
            if line.startswith('print("\\n" + "="*50)') or line.startswith('print("="*50)') or (line.startswith('print("EXPERIMENT') and '")' in line):
                continue
            code_lines.append(line)
        code_clean = '\n'.join(code_lines).strip()

        # Heading Table Banner
        heading_p = Paragraph(f"<b>{title}</b>", exp_heading_style)
        heading_table = Table([[heading_p]], colWidths=[532])
        heading_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))

        # Code block Preformatted inside Table
        code_p = Preformatted(code_clean, code_style)
        code_table = Table([[code_p]], colWidths=[532])
        code_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))

        # Add to story with KeepTogether where appropriate
        exp_flowables = [heading_table, code_table, Spacer(1, 14)]
        story.append(KeepTogether(exp_flowables))

    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build successful: Machine_Learning_Lab_Manual.pdf")

if __name__ == '__main__':
    build_pdf()
