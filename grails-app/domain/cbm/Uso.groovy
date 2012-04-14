package cbm

class Uso {
	static belongsTo = [indicaciones:Indicaciones, dosis:Dosis, medicamento:Medicamento]
    static constraints = {
		medicamento unique:["indicaciones"]
    }
	String toString(){""+indicaciones+":"+dosis}
}
