"""
Authors:
- zsolnai - georg.zsolnai123@gmail.com

Usage: groupOneApp/views.py
"""

from django import template

register = template.Library()

@register.filter(name='addclass')
def addclass(value, arg):
    return value.as_widget(attrs={'class': arg})