package cbm



import org.junit.*
import grails.test.mixin.*

@TestFor(EspecialidadController)
@Mock(Especialidad)
class EspecialidadControllerTests {


    def populateValidParams(params) {
      assert params != null
      // TODO: Populate valid properties like...
      //params["name"] = 'someValidName'
    }

    void testIndex() {
        controller.index()
        assert "/especialidad/list" == response.redirectedUrl
    }

    void testList() {

        def model = controller.list()

        assert model.especialidadInstanceList.size() == 0
        assert model.especialidadInstanceTotal == 0
    }

    void testCreate() {
       def model = controller.create()

       assert model.especialidadInstance != null
    }

    void testSave() {
        controller.save()

        assert model.especialidadInstance != null
        assert view == '/especialidad/create'

        response.reset()

        populateValidParams(params)
        controller.save()

        assert response.redirectedUrl == '/especialidad/show/1'
        assert controller.flash.message != null
        assert Especialidad.count() == 1
    }

    void testShow() {
        controller.show()

        assert flash.message != null
        assert response.redirectedUrl == '/especialidad/list'


        populateValidParams(params)
        def especialidad = new Especialidad(params)

        assert especialidad.save() != null

        params.id = especialidad.id

        def model = controller.show()

        assert model.especialidadInstance == especialidad
    }

    void testEdit() {
        controller.edit()

        assert flash.message != null
        assert response.redirectedUrl == '/especialidad/list'


        populateValidParams(params)
        def especialidad = new Especialidad(params)

        assert especialidad.save() != null

        params.id = especialidad.id

        def model = controller.edit()

        assert model.especialidadInstance == especialidad
    }

    void testUpdate() {
        controller.update()

        assert flash.message != null
        assert response.redirectedUrl == '/especialidad/list'

        response.reset()


        populateValidParams(params)
        def especialidad = new Especialidad(params)

        assert especialidad.save() != null

        // test invalid parameters in update
        params.id = especialidad.id
        //TODO: add invalid values to params object

        controller.update()

        assert view == "/especialidad/edit"
        assert model.especialidadInstance != null

        especialidad.clearErrors()

        populateValidParams(params)
        controller.update()

        assert response.redirectedUrl == "/especialidad/show/$especialidad.id"
        assert flash.message != null

        //test outdated version number
        response.reset()
        especialidad.clearErrors()

        populateValidParams(params)
        params.id = especialidad.id
        params.version = -1
        controller.update()

        assert view == "/especialidad/edit"
        assert model.especialidadInstance != null
        assert model.especialidadInstance.errors.getFieldError('version')
        assert flash.message != null
    }

    void testDelete() {
        controller.delete()
        assert flash.message != null
        assert response.redirectedUrl == '/especialidad/list'

        response.reset()

        populateValidParams(params)
        def especialidad = new Especialidad(params)

        assert especialidad.save() != null
        assert Especialidad.count() == 1

        params.id = especialidad.id

        controller.delete()

        assert Especialidad.count() == 0
        assert Especialidad.get(especialidad.id) == null
        assert response.redirectedUrl == '/especialidad/list'
    }
}
