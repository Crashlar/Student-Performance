from flask import Flask , request , render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.studentperformance.utils import load_object 

application = Flask(__name__)


