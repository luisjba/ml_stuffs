package cbm

class Medicamento {
	String nombre
	String contraindicaciones
	static belongsTo = [especialidad:Especialidad]
	static hasMany = [presentaciones:Presentacion, usos:Uso]
    static constraints = {
		nombre unique:true, blank:false
		contraindicaciones widget:"textarea"
    }
	String toString(){nombre}
}
