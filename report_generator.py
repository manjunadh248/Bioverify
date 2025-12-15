"""
BioVerify PDF Report Generator
Generates verification reports in PDF format
"""

import os
from datetime import datetime
from io import BytesIO

# Try to import reportlab, use fallback if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(verification_data, output_path=None):
    """
    Generate a PDF verification report.
    
    Args:
        verification_data: dict with keys:
            - username: str
            - account_score: float
            - liveness_score: float
            - verdict: str
            - timestamp: str
            - details: dict (optional)
        output_path: str (optional) - path to save PDF
    
    Returns:
        BytesIO buffer if output_path is None, else saves to file
    """
    
    if not REPORTLAB_AVAILABLE:
        return None
    
    # Create buffer or file
    if output_path:
        buffer = output_path
    else:
        buffer = BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a2e'),
        spaceBefore=20,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=8
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("🔒 BioVerify Verification Report", title_style))
    content.append(Spacer(1, 20))
    
    # Timestamp
    timestamp = verification_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    content.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", normal_style))
    content.append(Spacer(1, 20))
    
    # Account Information
    content.append(Paragraph("📋 Account Information", heading_style))
    
    username = verification_data.get('username', 'Unknown')
    content.append(Paragraph(f"<b>Username:</b> @{username}", normal_style))
    content.append(Spacer(1, 10))
    
    # Scores Table
    content.append(Paragraph("📊 Verification Scores", heading_style))
    
    account_score = verification_data.get('account_score', 0)
    liveness_score = verification_data.get('liveness_score', 0)
    combined_score = (account_score * 0.6 + (100 - liveness_score) * 0.4)
    
    score_data = [
        ['Metric', 'Score', 'Status'],
        ['Account Risk Score', f'{account_score:.1f}%', '⚠️ Risk' if account_score > 50 else '✅ OK'],
        ['Liveness Score', f'{liveness_score:.1f}%', '✅ Passed' if liveness_score >= 50 else '❌ Failed'],
        ['Combined Risk', f'{combined_score:.1f}%', '🚫 High' if combined_score > 60 else '✅ Low']
    ]
    
    score_table = Table(score_data, colWidths=[200, 100, 100])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 1, colors.gray),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    
    content.append(score_table)
    content.append(Spacer(1, 20))
    
    # Final Verdict
    content.append(Paragraph("🎯 Final Verdict", heading_style))
    
    verdict = verification_data.get('verdict', 'UNKNOWN')
    verdict_color = '#38ef7d' if verdict == 'REAL' else '#eb3349' if verdict == 'FAKE' else '#f4a460'
    
    verdict_style = ParagraphStyle(
        'Verdict',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor(verdict_color),
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=10
    )
    
    content.append(Paragraph(f"<b>{verdict}</b>", verdict_style))
    content.append(Spacer(1, 10))
    
    # Recommendation
    content.append(Paragraph("📝 Recommendation", heading_style))
    
    if verdict == 'REAL':
        recommendation = "This account demonstrates genuine characteristics. Safe to proceed with verification."
    elif verdict == 'FAKE':
        recommendation = "This account shows strong indicators of being fake or automated. Recommend blocking or further investigation."
    else:
        recommendation = "This account shows some suspicious patterns. Recommend manual review before approval."
    
    content.append(Paragraph(recommendation, normal_style))
    content.append(Spacer(1, 30))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.gray,
        alignment=TA_CENTER
    )
    content.append(Paragraph("Generated by BioVerify - Multi-Modal Fake Account Detection System", footer_style))
    content.append(Paragraph(f"Report ID: BV-{datetime.now().strftime('%Y%m%d%H%M%S')}", footer_style))
    
    # Build PDF
    doc.build(content)
    
    if isinstance(buffer, BytesIO):
        buffer.seek(0)
        return buffer
    
    return output_path


def generate_simple_report(verification_data):
    """Generate a simple text report if ReportLab is not available"""
    
    lines = []
    lines.append("=" * 50)
    lines.append("       BIOVERIFY VERIFICATION REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("ACCOUNT INFORMATION")
    lines.append("-" * 30)
    lines.append(f"Username: @{verification_data.get('username', 'Unknown')}")
    lines.append("")
    lines.append("VERIFICATION SCORES")
    lines.append("-" * 30)
    lines.append(f"Account Risk Score: {verification_data.get('account_score', 0):.1f}%")
    lines.append(f"Liveness Score: {verification_data.get('liveness_score', 0):.1f}%")
    lines.append("")
    lines.append("FINAL VERDICT")
    lines.append("-" * 30)
    lines.append(f"  {verification_data.get('verdict', 'UNKNOWN')}")
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test report generation
    test_data = {
        'username': 'test_user',
        'account_score': 35.5,
        'liveness_score': 85.0,
        'verdict': 'REAL',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if REPORTLAB_AVAILABLE:
        pdf = generate_pdf_report(test_data, 'test_report.pdf')
        print("PDF report generated: test_report.pdf")
    else:
        print("ReportLab not available. Text report:")
        print(generate_simple_report(test_data))
