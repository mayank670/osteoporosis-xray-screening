"""Server-side form validation (Flask-WTF adds a CSRF token to every form)."""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import BooleanField, EmailField, FloatField, IntegerField, PasswordField, SelectField, StringField
from wtforms.validators import DataRequired, EqualTo, Length, NumberRange, Optional, Regexp

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def strip_lower(value: str | None) -> str | None:
    """Filters run BEFORE validators: ' Bob@Example.com ' -> 'bob@example.com'."""
    return value.strip().lower() if value else value


class RegistrationForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(max=100)])
    email = EmailField("Email", filters=[strip_lower], validators=[
        DataRequired(), Length(max=255), Regexp(EMAIL_PATTERN, message="Enter a valid email address.")])
    password = PasswordField("Password", validators=[
        DataRequired(), Length(min=8, max=128, message="Password must be 8-128 characters.")])
    confirm = PasswordField("Confirm password", validators=[
        DataRequired(), EqualTo("password", message="Passwords do not match.")])


class LoginForm(FlaskForm):
    email = EmailField("Email", filters=[strip_lower], validators=[DataRequired(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=128)])


class UploadForm(FlaskForm):
    image = FileField("Knee X-ray image", validators=[FileRequired(message="Choose an image to upload.")])

    # Optional patient details -> hybrid (X-ray + clinical) model.
    # Ranges roughly match the training population (ages 17-107, adults only).
    # Ticked by default: the X-ray alone is close to chance (docs/model_training.md).
    # On a submitted form an unticked box is simply absent, so data becomes False.
    use_clinical = BooleanField("Add patient details (recommended — much more reliable)", default=True)
    age = IntegerField("Age (years)", validators=[Optional(), NumberRange(18, 110)])
    sex = SelectField("Sex", choices=[("", "Select…"), ("female", "Female"), ("male", "Male")],
                      validators=[Optional()])
    height_cm = FloatField("Height (cm)", validators=[Optional(), NumberRange(100, 220)])
    weight_kg = FloatField("Weight (kg)", validators=[Optional(), NumberRange(25, 250)])
    menopause_age = IntegerField("Age at menopause (leave empty if not post-menopausal)",
                                 validators=[Optional(), NumberRange(25, 65)])
    walking_km = FloatField("Maximum walking distance (km, optional)", validators=[Optional(), NumberRange(0, 50)])
    smoker = BooleanField("Smoker")
    diabetic = BooleanField("Diabetes")
    hypothyroidism = BooleanField("Hypothyroidism")
    estrogen_use = BooleanField("Oestrogen therapy")
    fracture_history = BooleanField("Previous fracture")
    family_history = BooleanField("Family history of osteoporosis")

    CORE_FIELDS = ("age", "sex", "height_cm", "weight_kg")
    FLAG_FIELDS = ("smoker", "diabetic", "hypothyroidism", "estrogen_use", "fracture_history", "family_history")

    def validate(self, extra_validators=None) -> bool:
        valid = super().validate(extra_validators)
        if not self.use_clinical.data:
            return valid
        for name in self.CORE_FIELDS:
            field = getattr(self, name)
            if field.data in (None, ""):
                field.errors.append("Required when patient details are included.")
                valid = False
        if self.menopause_age.data is not None:
            if self.sex.data == "male":
                self.menopause_age.errors.append("Only applies to female patients.")
                valid = False
            elif self.age.data is not None and self.menopause_age.data > self.age.data:
                self.menopause_age.errors.append("Cannot be later than the current age.")
                valid = False
        return valid

    def clinical_inputs(self) -> dict | None:
        """The patient details as a plain dict, or None if not included."""
        if not self.use_clinical.data:
            return None
        inputs = {"age": self.age.data, "sex": self.sex.data, "height_cm": self.height_cm.data,
                  "weight_kg": self.weight_kg.data, "menopause_age": self.menopause_age.data,
                  "walking_km": self.walking_km.data}
        inputs.update({name: bool(getattr(self, name).data) for name in self.FLAG_FIELDS})
        return inputs


class LogoutForm(FlaskForm):
    """Empty form: exists only to carry a CSRF token for the POST /logout button."""
